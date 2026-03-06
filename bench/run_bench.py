# run_bench.py
# Streamlined benchmarking for Nuisance Novelty paper.
#
# METRIC DEFINITIONS (canonical for the entire pipeline):
#   CSA  = N_correct / N_known                              (closed-set accuracy)
#   CCR  = (Correct & Accepted) / N_known                   (correct classification rate @ theta)
#   NNR  = Nuisance_Novelty / N_known                       (nuisance novelty prevalence)
#   URR  = rejected_unknowns / N_unknown                    (unknown rejection rate @ theta)
#   OSA  = (Correct&Accepted + rejected_unknowns) / (N_known + N_unknown)

from __future__ import annotations

import argparse
import gc
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

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
from bench.metrics import compute_auoscr, compute_ood_detection_metrics


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# -------------------------
# Utilities
# -------------------------
def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def safe_get_loader(dataset_name: str, batch_size: int = 64):
    """Load dataset, trying alternate name formats if needed."""
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

        results["closed_set_acc"] = n_correct / n_known
        results["ccr"] = (correct & accepted).sum().item() / n_known
        results["nnr"] = nn_count / n_known
    else:
        results["closed_set_acc"] = float("nan")
        results["ccr"] = float("nan")
        results["nnr"] = float("nan")

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
# STREAMING Inference
# -------------------------
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
    Run inference and return results.
    Returns: (labels, predictions, confidences, metadata)
    """
    all_labels = []
    all_preds = []
    all_confs = []
    all_metadata = []

    model.eval()
    det_name = getattr(detector, "bench_name", "").lower()
    # ODIN requires gradients for input perturbation
    needs_grad = det_name in {"gradnorm", "odin"}

    for batch in tqdm(loader, desc=desc, leave=False):
        if batch is None:
            continue

        img = batch["data"].to(device, non_blocking=True)
        labels = batch["label"]

        with torch.no_grad():
            with autocast(enabled=use_amp):
                logits = model(img)
            preds = logits.argmax(dim=1)

        if needs_grad:
            # ODIN/GradNorm need gradients - postprocess handles requires_grad internally
            model.zero_grad(set_to_none=True)
            _, confs = detector.postprocess(model, img)
        else:
            with torch.no_grad():
                _, confs = detector.postprocess(model, img)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_confs.append(confs.cpu())

        if store_metadata:
            bs = len(labels)
            for i in range(bs):
                all_metadata.append({
                    "path": batch["path"][i],
                    "level": batch["level"][i].item() if torch.is_tensor(batch["level"][i]) else batch["level"][i],
                    "nuisance": batch["nuisance"][i],
                    "dataset_name": batch["dataset_name"][i],
                })

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
# Main Benchmarking Loop
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run OOD benchmark. Config file provides all settings."
    )
    parser.add_argument("--config", required=True,
                        help="Path to config YAML (e.g., bench/configs/imagenet.yaml)")
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Enable mixed precision")
    parser.add_argument("--force_cpu", action="store_true", default=False,
                        help="Force CPU mode")
    parser.add_argument("--skip_samples", action="store_true", default=False,
                        help="Skip per-sample CSV output")
    parser.add_argument("--skip_ood_samples", action="store_true", default=False,
                        help="Skip per-sample OOD predictions (saves disk space; needed for exact ensemble OSA)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    bench_cfg = cfg.get("benchmark", {})

    # Extract settings from config
    backbones = bench_cfg.get("backbones", ["resnet50", "vit_b_16", "swin_t", "densenet121", "convnext_t"])
    detectors = bench_cfg.get("detectors", ["msp", "odin", "react", "ash", "dice", "knn", "mds", "vim", "she", "ebo", "postmax"])
    id_dataset = bench_cfg.get("id_dataset", "ImageNet-Val")
    train_dataset = bench_cfg.get("train_dataset", "ImageNet-Train")
    test_datasets = bench_cfg.get("test_datasets", ["ImageNet-LN"])
    # Support both single string (legacy) and list of OOD datasets
    test_ood_datasets_cfg = bench_cfg.get("test_ood_datasets", bench_cfg.get("test_ood_dataset", "OpenImage-O-Test"))
    if isinstance(test_ood_datasets_cfg, str):
        test_ood_datasets = [test_ood_datasets_cfg]
    else:
        test_ood_datasets = test_ood_datasets_cfg
    calib_id_dataset = bench_cfg.get("calib_id_dataset", "ImageNet-Val")
    calib_ood_dataset = bench_cfg.get("calib_ood_dataset", "OpenImage-O-Surrogate")
    batch_size = bench_cfg.get("batch_size", 32)
    out_dir = bench_cfg.get("out_dir", "analysis/bench_results")
    semantic_ood_markers = bench_cfg.get("semantic_ood_markers", ["openimage"])

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

    print(f"[CONFIG] {args.config}")
    print(f"[CONFIG] Backbones: {backbones}")
    print(f"[CONFIG] Detectors: {detectors}")
    print(f"[CONFIG] Test datasets: {test_datasets}")
    print(f"[CONFIG] Test OOD datasets: {test_ood_datasets}")

    os.makedirs(out_dir, exist_ok=True)

    summary_rows: List[Dict] = []
    all_samples: List[Dict] = []
    all_ood_samples: List[Dict] = []  # Per-sample OOD predictions for ensemble OSA

    # ========================================
    # Main Loop
    # ========================================
    for bb_name in backbones:
        print(f"\n{'=' * 60}\nBackbone: {bb_name}\n{'=' * 60}")

        base_model = load_backbone_from_ln_config(bb_name, device, args.config)

        for det_name in detectors:
            print(f"\n--- Detector: {det_name} ---")

            detector = get_detector(det_name, dataset_name=id_dataset)

            # Setup detector
            if requires_train_loader(det_name.lower()):
                print(f"  [BUILD] Extracting training features...")
                train_loader = safe_get_loader(train_dataset, batch_size)
                id_eval_loader = safe_get_loader(id_dataset, batch_size)
                detector.setup(base_model, {"train": train_loader, "val": id_eval_loader}, None)
                del train_loader
                clear_memory()
            else:
                id_eval_loader = safe_get_loader(id_dataset, batch_size)
                detector.setup(base_model, {"val": id_eval_loader}, None)

            # ----------------------------------------
            # Threshold Calibration
            # ----------------------------------------
            print(f"  [CALIB] Computing threshold...")
            calib_id_loader = safe_get_loader(calib_id_dataset, batch_size)
            calib_ood_loader = safe_get_loader(calib_ood_dataset, batch_size)

            gt_id, pred_id, conf_id, _ = run_inference_streaming(
                base_model, detector, calib_id_loader, device,
                desc="Calib-ID", use_amp=args.use_amp, store_metadata=False
            )
            gt_ood, pred_ood, conf_ood, _ = run_inference_streaming(
                base_model, detector, calib_ood_loader, device,
                desc="Calib-OOD", use_amp=args.use_amp, store_metadata=False
            )

            gt_calib = torch.cat([gt_id, torch.full_like(gt_ood, -1)])
            pred_calib = torch.cat([pred_id, pred_ood])
            conf_calib = torch.cat([conf_id, conf_ood])

            oosa_threshold = osa_mod.OSA(gt_calib, pred_calib, conf_calib, thresh=None, algo_name=det_name)
            _, val_score = osa_mod.OSA(gt_calib, pred_calib, conf_calib, thresh=oosa_threshold, algo_name=det_name)
            print(f"  Threshold: {oosa_threshold:.4f} (Val OSA: {val_score:.4f})")

            del gt_id, pred_id, conf_id
            del gt_calib, pred_calib, conf_calib
            del calib_id_loader, calib_ood_loader
            clear_memory()

            # ----------------------------------------
            # Loop over OOD test datasets
            # ----------------------------------------
            for test_ood_dataset in test_ood_datasets:
                print(f"  [TEST-OOD] {test_ood_dataset}")
                test_ood_loader = safe_get_loader(test_ood_dataset, batch_size)
                # Store metadata for OOD samples (needed for exact ensemble OSA)
                gt_ood, pred_ood, conf_ood, ood_metadata = run_inference_streaming(
                    base_model, detector, test_ood_loader, device,
                    desc=f"Test-OOD-{test_ood_dataset}", use_amp=args.use_amp,
                    store_metadata=not args.skip_ood_samples
                )

                n_ood = len(gt_ood)
                rejected_unknowns = (conf_ood < oosa_threshold).sum().item()
                urr_ood = rejected_unknowns / n_ood if n_ood > 0 else float("nan")
                print(f"      URR@t: {urr_ood:.4f} (N={n_ood}, rejected={rejected_unknowns})")

                # Save per-sample OOD predictions for exact ensemble OSA
                if not args.skip_ood_samples and ood_metadata:
                    for i, meta in enumerate(ood_metadata):
                        conf = conf_ood[i].item()
                        is_rejected = conf < oosa_threshold
                        all_ood_samples.append({
                            "backbone": bb_name,
                            "detector": det_name,
                            "test_ood_dataset": test_ood_dataset,
                            "path": meta.get("path", ""),
                            "confidence": conf,
                            "threshold_used": oosa_threshold,
                            "is_rejected": int(is_rejected),
                        })

                del test_ood_loader
                if args.skip_ood_samples:
                    del ood_metadata
                clear_memory()

                # ----------------------------------------
                # Test each nuisance dataset
                # ----------------------------------------
                for test_name in test_datasets:
                    print(f"    [TEST] {test_name} vs {test_ood_dataset}")

                    test_loader = safe_get_loader(test_name, batch_size)
                    gt_test, pred_test, conf_test, metadata = run_inference_streaming(
                        base_model, detector, test_loader, device,
                        desc=test_name, use_amp=args.use_amp, store_metadata=not args.skip_samples
                    )

                    semantic_ood = is_semantic_ood_dataset(test_name, semantic_ood_markers)

                    if semantic_ood:
                        gt_test[:] = -1

                    if not semantic_ood:
                        gt_combined = torch.cat([gt_test, torch.full((n_ood,), -1, dtype=torch.long)])
                        pred_combined = torch.cat([pred_test, pred_ood])
                        conf_combined = torch.cat([conf_test, conf_ood])
                        metrics = compute_metrics_batch(gt_combined, pred_combined, conf_combined, oosa_threshold)
                        metrics["urr"] = urr_ood
                        metrics["rejected_unknowns"] = rejected_unknowns

                        # Compute AUOSCR (COSTARR protocol metric)
                        auoscr = compute_auoscr(
                            gt_test.numpy(),
                            pred_test.numpy(),
                            conf_test.numpy(),
                            conf_ood.numpy()
                        )
                        metrics["auoscr"] = auoscr

                        # Compute AUROC and FPR@95TPR
                        ood_metrics = compute_ood_detection_metrics(conf_test.numpy(), conf_ood.numpy())
                        metrics["auroc"] = ood_metrics["AUROC"]
                        metrics["fpr_at_95tpr"] = ood_metrics["FPR@95TPR"]
                    else:
                        metrics = compute_metrics_batch(gt_test, pred_test, conf_test, oosa_threshold)
                        metrics["auoscr"] = float("nan")
                        metrics["auroc"] = float("nan")
                        metrics["fpr_at_95tpr"] = float("nan")

                    auoscr_str = f" AUOSCR={metrics['auoscr']:.4f}" if not np.isnan(metrics.get('auoscr', float('nan'))) else ""
                    print(f"        OSA={metrics['osa']:.4f} CSA={metrics['closed_set_acc']:.4f} "
                          f"CCR={metrics['ccr']:.4f} URR={metrics['urr']:.4f} NNR={metrics['nnr']:.4f}{auoscr_str}")

                    summary_rows.append({
                        "backbone": bb_name,
                        "detector": det_name,
                        "test_dataset": test_name,
                        "test_ood_dataset": test_ood_dataset,
                        "oosa_threshold": float(oosa_threshold),
                        "OSA": float(metrics["osa"]),
                        "CSA": float(metrics["closed_set_acc"]),
                        "CCR@theta": float(metrics["ccr"]),
                        "URR@theta": float(metrics["urr"]),
                        "NNR": float(metrics["nnr"]),
                        "AUOSCR": float(metrics.get("auoscr", float("nan"))),
                        "AUROC": float(metrics.get("auroc", float("nan"))),
                        "FPR@95TPR": float(metrics.get("fpr_at_95tpr", float("nan"))),
                        "n_known": metrics["n_known"],
                        "n_unknown": metrics["n_unknown"],
                        "rejected_unknowns": int(metrics["rejected_unknowns"]),
                    })

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
                                "n_unknown": n_ood,
                                "rejected_unknowns": rejected_unknowns,
                                "test_ood_dataset": test_ood_dataset,
                            })

                    del test_loader, gt_test, pred_test, conf_test, metadata
                    clear_memory()

                del gt_ood, pred_ood, conf_ood
                clear_memory()

        del base_model
        clear_memory()

    # ========================================
    # Save Results
    # ========================================
    print("\n[SAVE] Writing results...")
    pd.DataFrame(summary_rows).to_csv(f"{out_dir}/bench_summary.csv", index=False)

    if not args.skip_samples and all_samples:
        pd.DataFrame(all_samples).to_csv(f"{out_dir}/bench_samples.csv", index=False)

    if not args.skip_ood_samples and all_ood_samples:
        pd.DataFrame(all_ood_samples).to_csv(f"{out_dir}/ood_samples.csv", index=False)
        print(f"  OOD samples: {len(all_ood_samples)} rows")

    print(f"Done! Results in {out_dir}/")


if __name__ == "__main__":
    main()
