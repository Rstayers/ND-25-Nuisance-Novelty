# run_bench.py  (with caching)
from pathlib import Path

import torch
import argparse
import os
import pandas as pd
import numpy as np
import json
import hashlib

from tqdm import tqdm

from bench.detectors import get_detector, requires_train_loader
from bench.loader import get_loader
from bench.backbones import load_backbone
from bench import OSA as osa_mod

CACHE_DIR = "cache"
CACHE_PATH = os.path.join(CACHE_DIR, "oosa_thresholds.json")


def _hash_loader(loader):
    """Small hash to detect calibration-set changes (num images + dataset name)."""
    try:
        n = len(loader.dataset)
        name = getattr(loader.dataset, "name", "unknown")
        return hashlib.md5(f"{name}_{n}".encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


def _save_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def compute_id_threshold(confidences, tpr=0.95):
    if len(confidences) == 0:
        return 0.0
    confidences = np.sort(confidences)
    cutoff_index = int(len(confidences) * (1 - tpr))
    return confidences[cutoff_index]


def get_outcome(is_correct, confidence, threshold):
    is_accepted = confidence >= threshold
    if is_correct and is_accepted:
        return "Clean_Success"
    if is_correct and not is_accepted:
        return "Nuisance_Novelty"
    if not is_correct and not is_accepted:
        return "Double_Failure"
    if not is_correct and is_accepted:
        return "Contained_Misidentification"
    return "Error"


class CachedForwardModel(torch.nn.Module):
    def __init__(self, base_model, cached_logits, cached_feature):
        super().__init__()
        self.base_model = base_model
        self._cached_logits = cached_logits
        self._cached_feature = cached_feature

    def __getattr__(self, name):
        if name in {"base_model", "_cached_logits", "_cached_feature"}:
            return super().__getattr__(name)
        return getattr(self.base_model, name)

    def forward(self, x, return_feature=False, return_feature_list=False):
        if return_feature:
            return self._cached_logits, self._cached_feature
        return self._cached_logits

    def get_features(self, x):
        return self._cached_feature

    def forward_threshold(self, x, threshold):
        feat = self._cached_feature.clip(max=threshold)
        return self.base_model.fc(feat)


def run_inference(model, postprocessor, loader, device, desc=""):
    results = []
    print(f"Running Inference: {desc}")
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if batch is None:
                continue
            img = batch["data"].to(device)
            labels = batch["label"].to(device)

            # Forward pass to get logits + features
            logits, feat = model(img, return_feature=True)
            clean_preds = logits.argmax(dim=1)

            # ======== FIX: Select inference mode per detector ========
            det_name = getattr(postprocessor, "bench_name", "").lower()

            # Detectors needing raw model (require grad/features): GradNorm, KNN, MDS, DICE
            if det_name in ["gradnorm", "knn", "mds", "dice"]:
                # Call postprocessor directly on the real model (no caching)
                _, confs = postprocessor.postprocess(model, img)
            else:
                # Use cached logits/features to save time for MSP/MaxLogit/React/EBO/etc.
                cached_model = CachedForwardModel(model, logits, feat)
                _, confs = postprocessor.postprocess(cached_model, img)
            # ========================================================

            preds = clean_preds.cpu().numpy()
            confs = confs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            bs = len(preds)
            for i in range(bs):
                results.append({
                    "dataset": batch["dataset_name"][i],
                    "path": batch["path"][i],
                    "label": labels_np[i],
                    "prediction": preds[i],
                    "confidence": confs[i],
                    "correct_cls": int(preds[i] == labels_np[i]),
                    "level": batch["level"][i].item(),
                    "nuisance": batch["nuisance"][i],
                    "competency_parce": batch["parce"][i].item(),
                })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["resnet50", "vit_b_16", "densenet121", "swin_t", "convnext_t"],
    )
    parser.add_argument("--detectors", nargs="+", default=["msp", "react", "ebo"])
    parser.add_argument("--id_dataset", default="Cars-Test")
    parser.add_argument("--test_datasets", nargs="+", default=["Cars_LN"])
    parser.add_argument("--out_dir", default="analysis/bench_results/cars")

    # --- NEW ARGUMENTS FOR CALIBRATION DATASETS ---
    parser.add_argument("--calib_id_dataset", default="Cars-Test",
                        help="ID dataset used for OSA calibration (default: ImageNetV2-Val)")
    parser.add_argument("--calib_ood_dataset", default="OpenImage-O-Surrogate",
                        help="OOD dataset used for OSA calibration (default: OpenImage-O-Surrogate)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = _load_cache()

    # Load ID loaders (needed for Detector setup like KNN/MDS)
    # For Cars, we probably don't have a specific 'Train' loader setup in datasets yet,
    # so these might default to ImageNet if not careful.
    # If you want KNN on cars, you must add 'Stanford-Cars-Train' to datasets.py.
    id_loader_val = get_loader(args.id_dataset, batch_size=64)
    # Only load training set if a detector needs it
    id_loader_train = None

    print(f"Loading OOSA calibration sets ({args.calib_id_dataset}, {args.calib_ood_dataset}).")
    id_val_loader = get_loader(args.calib_id_dataset, batch_size=64)
    surrogate_loader = get_loader(args.calib_ood_dataset, batch_size=64)
    calib_hash = f"{_hash_loader(id_val_loader)}_{_hash_loader(surrogate_loader)}"

    test_loaders = []
    for name in args.test_datasets:
        print(f"Loading Test Dataset: {name}")
        test_loaders.append(get_loader(name))

    all_rows = []

    for bb_name in args.backbones:
        print(f"\n=== Backbone: {bb_name} ===")

        # --- FIX: Pass the ID Dataset Name ---
        model = load_backbone(bb_name, device, dataset_name=args.id_dataset)

        for det_name in args.detectors:
            print(f"--- Detector: {det_name} ---")
            detector = get_detector(det_name)

            # Setup Detector
            if requires_train_loader(det_name):
                # Lazy load train loader only if needed
                if id_loader_train is None:
                    # You might need to add 'Stanford-Cars-Train' to datasets.py for this to work perfectly
                    id_loader_train = get_loader("Stanford-Cars-Train", batch_size=64)

                cache_path = Path(f"cache/features_{bb_name}_{det_name}_{args.id_dataset}.pt")
                if cache_path.exists():
                    print(f"[CACHE] Loading precomputed training features.")
                    feat_cache = torch.load(cache_path, map_location="cpu")
                    detector.feature_bank = feat_cache["feature_bank"]
                    detector.label_bank = feat_cache["label_bank"]
                    detector.is_fitted = True
                else:
                    print(f"[BUILD] Extracting training features for {det_name}...")
                    detector.setup(model, {"train": id_loader_train, "val": id_loader_val}, None)
                    if hasattr(detector, "feature_bank"):
                        torch.save({
                            "feature_bank": detector.feature_bank.cpu(),
                            "label_bank": detector.label_bank.cpu()
                        }, cache_path)
            else:
                detector.setup(model, {"val": id_loader_val}, None)

            # --- Calibration & Inference Loop (Same as your code) ---
            cache_key = f"{bb_name}|{det_name}"
            # ... (Rest of your OSA logic remains identical) ...

            # Re-implementing just the OSA logic block for clarity
            if cache_key in cache and cache[cache_key]["calib_hash"] == calib_hash:
                oosa_threshold = cache[cache_key]["threshold"]
                val_score = cache[cache_key]["val_score"]
                print(f"[CACHE] Using cached OOSA threshold ({oosa_threshold:.4f})")
            else:
                print(f"[{det_name}] --- OOSA Calibration ---")
                id_results = run_inference(model, detector, id_val_loader, device, desc=f"Calib-ID")
                surr_results = run_inference(model, detector, surrogate_loader, device, desc=f"Calib-OOD")

                gt_val = torch.tensor([r["label"] for r in id_results] + [r["label"] for r in surr_results])
                pred_val = torch.tensor([r["prediction"] for r in id_results] + [r["prediction"] for r in surr_results])
                prob_val = torch.tensor([r["confidence"] for r in id_results] + [r["confidence"] for r in surr_results],
                                        dtype=torch.float)
                gt_val[len(id_results):] = -1

                oosa_threshold = osa_mod.OSA(gt_val, pred_val, prob_val, thresh=None, algo_name=det_name)
                (_, _, _, _), val_score = osa_mod.OSA(gt_val, pred_val, prob_val, thresh=oosa_threshold,
                                                      algo_name=det_name)
                print(f" > OSA Threshold: {oosa_threshold:.4f}")

                cache[cache_key] = {"threshold": float(oosa_threshold), "val_score": float(val_score),
                                    "calib_hash": calib_hash}
                _save_cache(cache)

            for loader in test_loaders:
                dataset_name = loader.dataset.name
                results = run_inference(model, detector, loader, device, desc=dataset_name)

                gt_test = torch.tensor([r["label"] for r in results])
                prob_test = torch.tensor([r["confidence"] for r in results], dtype=torch.float)
                pred_test = torch.tensor([r["prediction"] for r in results])

                # OOD Logic for OSA calculation
                if any(x in dataset_name for x in ["OpenImage", "OOD", "CNS", "ImageNet-O"]):
                    gt_test[:] = -1

                n_known = (gt_test >= 0).sum().item()
                n_unknown = (gt_test < 0).sum().item()

                if n_known > 0 and n_unknown > 0:
                    _, osa_val = osa_mod.OSA(gt_test, pred_test, prob_test, thresh=oosa_threshold,
                                             algo_name=f"{det_name}_{dataset_name}")
                elif n_known == 0 and n_unknown > 0:
                    osa_val = (prob_test < oosa_threshold).sum().item() / n_unknown
                elif n_unknown == 0 and n_known > 0:
                    osa_val = ((prob_test >= oosa_threshold) & (gt_test == pred_test)).sum().item() / n_known
                else:
                    osa_val = np.nan

                print(f"[{det_name}] OSA({dataset_name}) = {osa_val:.4f}")

                for row in results:
                    row.update({
                        "backbone": bb_name,
                        "detector": det_name,
                        "threshold_used": oosa_threshold,
                        "OSA_dataset": osa_val,
                        "outcome": get_outcome(bool(row["correct_cls"]), row["confidence"], oosa_threshold)
                    })
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    # (Existing DataFrame saving logic)
    csv_path = os.path.join(args.out_dir, "benchmark_cars.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")
    _save_cache(cache)


if __name__ == "__main__":
    main()