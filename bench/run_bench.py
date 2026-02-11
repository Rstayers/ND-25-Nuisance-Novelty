# run_bench.py
# Memory-efficient benchmarking with full metric reconstruction support.
#
# METRIC DEFINITIONS (canonical for the entire pipeline):
#   CSA  = N_correct / N_known                              (closed-set accuracy)
#   CCR  = (Correct & Accepted) / N_known                   (correct classification rate @ theta)
#   NNR  = Nuisance_Novelty / N_known                       (nuisance novelty prevalence)
#   CNR  = Nuisance_Novelty / N_correct                     (conditional novelty rate on correct samples)
#   URR  = rejected_unknowns / N_unknown                    (unknown rejection rate @ theta)
#   OSA  = (Correct&Accepted + rejected_unknowns) / (N_known + N_unknown)
#
# Per-sample CSV includes n_unknown and rejected_unknowns so analysis/processing.py
# can reconstruct OSA, URR at any (backbone, detector, dataset, level) granularity.

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
import argparse
import gc
import os
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys

# Handle autocast compatibility
try:
    from torch.amp import autocast as _autocast


    def autocast(enabled=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return _autocast(device_type=device, enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast

from tqdm import tqdm

from bench.detectors import get_detector, requires_train_loader
from bench.loader import get_loader
from bench.backbones import load_backbone_from_ln_config
from bench import OSA as osa_mod
from bench.metrics import compute_ood_det_metrics

# -------------------------
# Config
# -------------------------
CACHE_DIR = "cache"
CACHE_PATH = os.path.join(CACHE_DIR, "oosa_thresholds.json")


# -------------------------
# Utilities
# -------------------------
def _hash_loader(loader) -> str:
    try:
        n = len(loader.dataset)
        name = getattr(loader.dataset, "name", "unknown")
        return hashlib.md5(f"{name}_{n}".encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


def _load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


def _save_cache(cache: Dict[str, Any]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def safe_get_loader(dataset_name: str, batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True):
    """Load dataset with memory-conscious defaults."""
    try:
        return get_loader(dataset_name, batch_size=batch_size)
    except Exception as e:
        alt = dataset_name.replace("_", "-")
        if alt != dataset_name:
            try:
                return get_loader(alt, batch_size=batch_size)
            except Exception:
                pass
        raise e


def infer_ln_config_path(id_dataset: str, override: Optional[str] = None) -> str:
    if override is not None and override != "":
        return override
    s = id_dataset.lower()
    if "imagenet" in s:
        return "ln_dataset/configs/imagenet.yaml"
    if "cars" in s:
        return "ln_dataset/configs/stanford_cars.yaml"
    if "cub" in s:
        return "ln_dataset/configs/cub.yaml"
    raise KeyError(f"Could not infer config from '{id_dataset}'")


def infer_train_dataset(id_dataset: str) -> str:
    s = id_dataset.lower()
    if "imagenet" in s:
        return "ImageNet-Train"
    if "cars" in s:
        return "Cars-Train"
    if "cub" in s:
        return "CUB-Train"
    return "ImageNet-Train"


def get_outcome(is_correct: bool, confidence: float, threshold: float) -> str:
    is_accepted = confidence >= threshold
    if is_correct and is_accepted:
        return "Clean_Success"
    if is_correct and not is_accepted:
        return "Nuisance_Novelty"
    if (not is_correct) and (not is_accepted):
        return "Double_Failure"
    return "Contained_Misidentification"


def is_semantic_ood_dataset(name: str, markers: List[str]) -> bool:
    low = name.lower()
    return any(m.lower() in low for m in markers)


# -------------------------
# Metric Helpers
# -------------------------
def compute_metrics_batch(
        gt: torch.Tensor,
        pred: torch.Tensor,
        conf: torch.Tensor,
        threshold: float
) -> Dict[str, float]:
    """
    Compute all metrics in one pass over combined known+unknown data.

    Returns dict with keys:
        n_known, n_unknown,
        closed_set_acc (CSA), ccr (CCR@theta),
        nnr (NN/N_known), cnr (NN/N_correct),
        urr (URR@theta), rejected_unknowns,
        osa (Open-Set Accuracy)
    """
    known_mask = gt >= 0
    unknown_mask = gt < 0
    n_known = known_mask.sum().item()
    n_unknown = unknown_mask.sum().item()

    results = {"n_known": n_known, "n_unknown": n_unknown}

    # --- Known-side metrics ---
    if n_known > 0:
        correct = (gt[known_mask] == pred[known_mask])
        accepted = (conf[known_mask] >= threshold)
        n_correct = correct.sum().item()
        nn_count = (correct & ~accepted).sum().item()

        results["closed_set_acc"] = n_correct / n_known               # CSA
        results["ccr"] = (correct & accepted).sum().item() / n_known   # CCR@theta
        results["nnr"] = nn_count / n_known                            # NNR = NN / N_known
        results["cnr"] = (nn_count / n_correct) if n_correct > 0 else float("nan")  # CNR = NN / N_correct
    else:
        results["closed_set_acc"] = float("nan")
        results["ccr"] = float("nan")
        results["nnr"] = float("nan")
        results["cnr"] = float("nan")

    # --- Unknown-side metrics ---
    if n_unknown > 0:
        rejected = (conf[unknown_mask] < threshold)
        results["rejected_unknowns"] = rejected.sum().item()
        results["urr"] = results["rejected_unknowns"] / n_unknown
    else:
        results["rejected_unknowns"] = 0
        results["urr"] = float("nan")

    # --- Joint metric ---
    if n_known > 0 and n_unknown > 0:
        correct_accepted = ((gt[known_mask] == pred[known_mask]) & (conf[known_mask] >= threshold)).sum().item()
        results["osa"] = (correct_accepted + results["rejected_unknowns"]) / (n_known + n_unknown)
    else:
        results["osa"] = float("nan")

    return results


# -------------------------
# STREAMING Inference (Low Memory)
# -------------------------
@torch.no_grad()
def run_inference_streaming(
        model: nn.Module,
        detector,
        loader,
        device: torch.device,
        desc: str = "",
        use_amp: bool = False,
        store_metadata: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Run inference and return results WITHOUT caching features.
    Returns: (labels, predictions, confidences, metadata)

    Memory efficient: processes batch by batch, only keeps final results.
    """
    all_labels = []
    all_preds = []
    all_confs = []
    all_metadata = []

    model.eval()
    det_name = getattr(detector, "bench_name", "").lower()
    needs_grad = det_name in {"gradnorm"}

    for batch in tqdm(loader, desc=desc, leave=False):
        if batch is None:
            continue

        img = batch["data"].to(device, non_blocking=True)
        labels = batch["label"]

        # Get predictions
        with autocast(enabled=use_amp):
            logits = model(img)
        preds = logits.argmax(dim=1)

        # Get confidence scores from detector
        if needs_grad:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                _, confs = detector.postprocess(model, img)
        else:
            _, confs = detector.postprocess(model, img)

        # Store results (CPU)
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_confs.append(confs.cpu())

        # Metadata (optional - can skip to save memory)
        if store_metadata:
            bs = len(labels)
            for i in range(bs):
                all_metadata.append({
                    "path": batch["path"][i],
                    "level": batch["level"][i].item() if torch.is_tensor(batch["level"][i]) else batch["level"][i],
                    "nuisance": batch["nuisance"][i],
                    "parce": batch["parce"][i].item() if torch.is_tensor(batch["parce"][i]) else batch["parce"][i],
                    "dataset_name": batch["dataset_name"][i],
                })

        # Clear GPU memory after each batch
        del img, logits, confs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        torch.cat(all_labels, dim=0),
        torch.cat(all_preds, dim=0),
        torch.cat(all_confs, dim=0),
        all_metadata
    )


# -------------------------
# Main Benchmarking Loop (Memory Efficient)
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=["swin_t", "resnet50", "vit_b_16", "densenet121", "convnext_t"])
    parser.add_argument("--detectors", nargs="+", default=[
            "odin",
         "react",
         "she",
            "dice",
            "postmax",
            "msp",

            "mds",
            "knn",
            "vim",


        ])
    parser.add_argument("--id_dataset", default="ImageNet-Val")
    parser.add_argument("--test_datasets", nargs="+", default=["ImageNet-LN", "CNS", "ImageNet-C"])
    parser.add_argument("--test_ood_dataset", default="OpenImage-O-Test")
    parser.add_argument("--calib_id_dataset", default="ImageNet-Val")
    parser.add_argument("--calib_ood_dataset", default="OpenImage-O-Surrogate")
    parser.add_argument("--batch_size", type=int, default=32, help="Smaller batch = less memory")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="analysis/bench_results/imagenet")
    parser.add_argument("--ln_config", default="")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--force_cpu", action="store_true", default=False)
    parser.add_argument("--skip_samples", action="store_true", default=False,
                        help="Skip per-sample CSV (saves memory)")
    parser.add_argument("--semantic_ood_markers", nargs="+",
                        default=["openimage", "imagenet-o", "textures", "inat", "sun", "places", "ninco"])
    args = parser.parse_args()

    # Device setup
    if args.force_cpu:
        device = torch.device("cpu")
        print("[INFO] Forced CPU mode")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("[INFO] Running on CPU")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    threshold_cache = _load_cache()

    summary_rows: List[Dict] = []
    all_samples: List[Dict] = []

    # ========================================
    # Main Loop
    # ========================================
    for bb_name in args.backbones:
        print(f"\n{'=' * 60}\nBackbone: {bb_name}\n{'=' * 60}")

        ln_cfg_path = infer_ln_config_path(args.id_dataset, override=args.ln_config)
        base_model = load_backbone_from_ln_config(bb_name, device, ln_cfg_path)

        for det_name in args.detectors:
            print(f"\n--- Detector: {det_name} ---")

            detector = get_detector(det_name, dataset_name=args.id_dataset)

            # Setup detector
            if requires_train_loader(det_name.lower()):
                train_name = infer_train_dataset(args.id_dataset)
                cache_path = Path(f"cache/features_{bb_name}_{det_name}_{args.id_dataset}.pt")

                if cache_path.exists():
                    print(f"  [CACHE] Loading training features")
                    feat_cache = torch.load(cache_path, map_location="cpu")
                    detector.feature_bank = feat_cache["feature_bank"]
                    detector.label_bank = feat_cache["label_bank"]
                    detector.is_fitted = True
                else:
                    print(f"  [BUILD] Extracting training features (one-time)...")
                    train_loader = safe_get_loader(train_name, args.batch_size, args.num_workers)
                    id_eval_loader = safe_get_loader(args.id_dataset, args.batch_size, args.num_workers)
                    detector.setup(base_model, {"train": train_loader, "val": id_eval_loader}, None)

                    if hasattr(detector, "feature_bank"):
                        torch.save({
                            "feature_bank": detector.feature_bank.cpu(),
                            "label_bank": detector.label_bank.cpu(),
                        }, cache_path)

                    del train_loader
                    clear_memory()
            else:
                id_eval_loader = safe_get_loader(args.id_dataset, args.batch_size, args.num_workers)
                detector.setup(base_model, {"val": id_eval_loader}, None)

            # ----------------------------------------
            # Threshold Calibration
            # ----------------------------------------
            calib_id_loader = safe_get_loader(args.calib_id_dataset, args.batch_size, args.num_workers)
            calib_ood_loader = safe_get_loader(args.calib_ood_dataset, args.batch_size, args.num_workers)
            calib_hash = f"{_hash_loader(calib_id_loader)}_{_hash_loader(calib_ood_loader)}"

            cache_key = f"{bb_name}|{det_name}"
            if cache_key in threshold_cache and threshold_cache[cache_key].get("calib_hash", "") == calib_hash:
                oosa_threshold = float(threshold_cache[cache_key]["threshold"])
                val_score = float(threshold_cache[cache_key].get("val_score", float("nan")))
                print(f"  [CACHE] Threshold: {oosa_threshold:.4f}")
            else:
                print(f"  [CALIB] Computing threshold...")

                # Run calibration inference
                gt_id, pred_id, conf_id, _ = run_inference_streaming(
                    base_model, detector, calib_id_loader, device,
                    desc="Calib-ID", use_amp=args.use_amp, store_metadata=False
                )
                gt_ood, pred_ood, conf_ood, _ = run_inference_streaming(
                    base_model, detector, calib_ood_loader, device,
                    desc="Calib-OOD", use_amp=args.use_amp, store_metadata=False
                )

                # Combine for OSA
                gt_calib = torch.cat([gt_id, torch.full_like(gt_ood, -1)])
                pred_calib = torch.cat([pred_id, pred_ood])
                conf_calib = torch.cat([conf_id, conf_ood])

                oosa_threshold = osa_mod.OSA(gt_calib, pred_calib, conf_calib, thresh=None, algo_name=det_name)
                _, val_score = osa_mod.OSA(gt_calib, pred_calib, conf_calib, thresh=oosa_threshold, algo_name=det_name)
                val_score = float(val_score)

                threshold_cache[cache_key] = {
                    "threshold": float(oosa_threshold),
                    "val_score": val_score,
                    "calib_hash": calib_hash,
                }
                _save_cache(threshold_cache)
                print(f"  [NEW] Threshold: {oosa_threshold:.4f} (Val OSA: {val_score:.4f})")

                # Clear calibration data
                del gt_id, pred_id, conf_id, gt_ood, pred_ood, conf_ood
                del gt_calib, pred_calib, conf_calib
                clear_memory()

            del calib_id_loader, calib_ood_loader
            clear_memory()

            # ----------------------------------------
            # Test-time OOD (for true OSA)
            # ----------------------------------------
            print(f"  [TEST-OOD] {args.test_ood_dataset}")
            test_ood_loader = safe_get_loader(args.test_ood_dataset, args.batch_size, args.num_workers)
            gt_ood, pred_ood, conf_ood, _ = run_inference_streaming(
                base_model, detector, test_ood_loader, device,
                desc="Test-OOD", use_amp=args.use_amp, store_metadata=False
            )

            # OOD-side constants: fixed for this (backbone, detector) pair
            n_ood = len(gt_ood)
            rejected_unknowns = (conf_ood < oosa_threshold).sum().item()
            urr_ood = rejected_unknowns / n_ood if n_ood > 0 else float("nan")
            print(f"      URR@\u03b8: {urr_ood:.4f} (N={n_ood}, rejected={rejected_unknowns})")

            del test_ood_loader
            clear_memory()

            # ----------------------------------------
            # ID Eval (for AUROC baseline)
            # ----------------------------------------
            id_eval_loader = safe_get_loader(args.id_dataset, args.batch_size, args.num_workers)
            gt_id, pred_id, conf_id, _ = run_inference_streaming(
                base_model, detector, id_eval_loader, device,
                desc=f"ID-Eval", use_amp=args.use_amp, store_metadata=False
            )
            id_scores = conf_id.numpy()

            del id_eval_loader
            clear_memory()

            # ----------------------------------------
            # Test each nuisance dataset
            # ----------------------------------------
            for test_name in args.test_datasets:
                print(f"  [TEST] {test_name}")

                test_loader = safe_get_loader(test_name, args.batch_size, args.num_workers)
                gt_test, pred_test, conf_test, metadata = run_inference_streaming(
                    base_model, detector, test_loader, device,
                    desc=test_name, use_amp=args.use_amp, store_metadata=not args.skip_samples
                )

                semantic_ood = is_semantic_ood_dataset(test_name, args.semantic_ood_markers)

                if semantic_ood:
                    gt_test[:] = -1

                # Combine with test OOD for true OSA
                if not semantic_ood:
                    gt_combined = torch.cat([gt_test, torch.full((n_ood,), -1, dtype=torch.long)])
                    pred_combined = torch.cat([pred_test, pred_ood])
                    conf_combined = torch.cat([conf_test, conf_ood])
                    metrics = compute_metrics_batch(gt_combined, pred_combined, conf_combined, oosa_threshold)
                    # Override with pre-computed OOD constants (identical, just cleaner)
                    metrics["urr"] = urr_ood
                    metrics["rejected_unknowns"] = rejected_unknowns
                else:
                    metrics = compute_metrics_batch(gt_test, pred_test, conf_test, oosa_threshold)

                # OOD detection metrics
                if semantic_ood:
                    ood_metrics = compute_ood_det_metrics(id_scores, conf_test.numpy(), tpr=0.95)
                else:
                    ood_metrics = {"AUROC": float("nan"), "AUPR_IN": float("nan"),
                                   "AUPR_OUT": float("nan"), "FPR@95TPR": float("nan")}

                print(f"      OSA={metrics['osa']:.4f} CSA={metrics['closed_set_acc']:.4f} "
                      f"CCR={metrics['ccr']:.4f} URR={metrics['urr']:.4f} "
                      f"NNR={metrics['nnr']:.4f} CNR={metrics['cnr']:.4f}")

                summary_rows.append({
                    "backbone": bb_name,
                    "detector": det_name,
                    "test_dataset": test_name,
                    "test_ood_dataset": args.test_ood_dataset,
                    "oosa_threshold": float(oosa_threshold),
                    # --- All metrics ---
                    "OSA": float(metrics["osa"]),
                    "CSA": float(metrics["closed_set_acc"]),
                    "CCR@theta": float(metrics["ccr"]),
                    "URR@theta": float(metrics["urr"]),
                    "NNR": float(metrics["nnr"]),
                    "CNR": float(metrics["cnr"]),
                    # --- OOD detection metrics ---
                    "AUROC": float(ood_metrics["AUROC"]),
                    "AUPR_IN": float(ood_metrics["AUPR_IN"]),
                    "AUPR_OUT": float(ood_metrics["AUPR_OUT"]),
                    "FPR@95TPR": float(ood_metrics["FPR@95TPR"]),
                    # --- Counts for reconstruction ---
                    "n_known": metrics["n_known"],
                    "n_unknown": metrics["n_unknown"],
                    "rejected_unknowns": int(metrics["rejected_unknowns"]),
                })

                # Per-sample records (optional)
                if not args.skip_samples and metadata:
                    for i, meta in enumerate(metadata):
                        is_correct = (pred_test[i] == gt_test[i]).item()
                        conf = conf_test[i].item()
                        all_samples.append({
                            **meta,
                            "label": gt_test[i].item(),
                            "prediction": pred_test[i].item(),
                            "confidence": conf,
                            "correct_cls": int(is_correct),
                            "backbone": bb_name,
                            "detector": det_name,
                            "threshold_used": oosa_threshold,
                            "outcome": get_outcome(is_correct, conf, oosa_threshold),
                            # --- OOD constants for per-level OSA reconstruction ---
                            "n_unknown": n_ood,
                            "rejected_unknowns": rejected_unknowns,
                        })

                del test_loader, gt_test, pred_test, conf_test, metadata
                clear_memory()

            # Clear OOD data
            del gt_ood, pred_ood, conf_ood, gt_id, pred_id, conf_id
            clear_memory()

        # Clear model between backbones
        del base_model
        clear_memory()

    # ========================================
    # Save Results
    # ========================================
    print("\n[SAVE] Writing results...")
    pd.DataFrame(summary_rows).to_csv(f"{args.out_dir}/bench_summary.csv", index=False)

    if not args.skip_samples and all_samples:
        pd.DataFrame(all_samples).to_csv(f"{args.out_dir}/bench_samples.csv", index=False)

    print(f"Done! Results in {args.out_dir}/")


if __name__ == "__main__":
    main()