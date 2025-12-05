# psycho_bench/main.py

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Any
import csv  # NEW: for streaming per-sample CSV writing

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from openood.evaluation_api import Evaluator

from backbones import load_psycho_backbones, build_backbone
from calibration import calibrate_thresholds_openood_style
from datasets import (
    DATASET_REGISTRY,
    DatasetSpec,
    build_loader_for_spec,
    build_id_shift_row,
    build_ood_row,
)
from metrics_summary import compute_ood_metrics, build_metrics_row



DEFAULT_DETECTORS = ["msp"]
DEFAULT_FPRS = [0.01, 0.05, 0.1]


def _require_cuda_or_exit():
    if not torch.cuda.is_available():
        print("[FATAL] CUDA is required by OpenOOD Evaluator. Exiting.",
              file=sys.stderr)
        sys.exit(1)


def _trial_id() -> str:
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:8]


def run_benchmark(
    data_root: str,
    out_dir: str,
    backbones: List[str],
    detectors: List[str],
    fprs: List[float],
    batch_size: int,
    num_workers: int,
    seed: int,
    dataset_names: List[str],
):
    """
    Unified psycho benchmark:

      - Runs ImageNet ID once per (backbone, detector) using OpenOOD Evaluator.
      - For each dataset in dataset_names (e.g. ['cns', 'imagenet_c', 'ninco']):
          * Runs inference.
          * Calibrates thresholds using ID + this dataset's scores (OpenOOD-style).
          * Logs per-sample rows for the dataset only (STREAMED to CSV).
          * Computes OpenOOD-style OOD metrics and logs a summary row.

    Outputs:
      - per-sample CSV for CNS + ImageNet-C + NINCO only (streamed, low memory).
      - metrics CSV with OpenOOD-style metrics per dataset.
      - thresholds JSON (per backbone / detector / dataset / FPR target).
    """
    _require_cuda_or_exit()
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"
    trial = _trial_id()

    # We now STREAM per-sample rows instead of keeping them in memory.
    per_sample_csv = os.path.join(out_dir, f"psycho_per_sample_{trial}.csv")
    sample_file = open(per_sample_csv, "w", newline="")
    sample_writer: csv.DictWriter | None = None
    sample_header_written = False

    metrics_rows: List[Dict[str, Any]] = []
    thresholds_log: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    # Discover psycho checkpoints
    psycho_ckpts = load_psycho_backbones("results")

    for backbone in backbones:
        print(f"\n========== Backbone: {backbone} ==========")
        thresholds_log[backbone] = {}

        for det in detectors:
            print(f"\n[Detector: {det}]")
            thresholds_log[backbone][det] = {}

            net = build_backbone(backbone, device, psycho_ckpts)

            evaluator = Evaluator(
                net=net,
                id_name=args.id_name,
                data_root=data_root,
                config_root="configs",
                preprocessor=None,
                postprocessor_name=det,
                postprocessor=None,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            # ----------------- ImageNet ID (for calibration + metrics) -----------------
            print("Running ImageNet ID inference (for calibration/metrics)...")
            id_loader = evaluator.dataloader_dict["id"]["test"]
            id_pred, id_conf, id_gt = evaluator.postprocessor.inference(
                evaluator.net,
                id_loader,
                progress=True,
            )
            id_pred = np.asarray(id_pred)
            id_conf = np.asarray(id_conf)
            id_gt = np.asarray(id_gt)
            n_id = len(id_conf)

            # Re-use transform for every dataset
            transform = None
            if hasattr(evaluator, "preprocessor") and hasattr(
                evaluator.preprocessor, "transform"
            ):
                transform = evaluator.preprocessor.transform

            # ----------------- Loop over datasets (CNS, ImageNet-C, NINCO, ...) -----------------
            for ds_name in dataset_names:
                if ds_name not in DATASET_REGISTRY:
                    print(f"[WARN] Dataset '{ds_name}' not in registry; skipping.")
                    continue

                spec: DatasetSpec = DATASET_REGISTRY[ds_name]
                print(f"\nRunning dataset: {ds_name} (role={spec.role})")

                loader = build_loader_for_spec(
                    spec=spec,
                    data_root=data_root,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    transform=transform,
                )

                preds, confs, gts, rel_paths = [], [], [], []
                net.eval()
                start_time = time.time()

                with torch.no_grad():
                    for imgs, labels, batch_rel_paths in tqdm(
                        loader, desc=f"{ds_name} [{backbone} | {det}]"
                    ):
                        imgs = imgs.to(device)
                        labels = labels.to(device)

                        # OpenOOD postprocessor expects (net, imgs)
                        batch_pred, batch_conf = evaluator.postprocessor.postprocess(
                            evaluator.net, imgs
                        )

                        preds.append(batch_pred.cpu().numpy())
                        confs.append(batch_conf.cpu().numpy())
                        gts.append(labels.cpu().numpy())
                        rel_paths.extend(list(batch_rel_paths))

                elapsed = time.time() - start_time
                ood_pred = np.concatenate(preds)
                ood_conf = np.concatenate(confs)
                ood_gt_raw = np.concatenate(gts)

                mean_conf = float(np.mean(ood_conf))
                print(
                    f"\n[{ds_name} Results] Backbone={backbone}, Detector={det} "
                    f"• Mean confidence: {mean_conf:.4f}, "
                    f"Time: {elapsed / 60:.2f} min for {len(loader.dataset):,} images"
                )

                # ----------------- Threshold calibration -----------------
                tau_map = calibrate_thresholds_openood_style(
                    id_conf=id_conf,
                    ood_conf=ood_conf,
                    fprs=fprs,
                )
                thresholds_log[backbone][det][ds_name] = {
                    str(k): float(v) for k, v in tau_map.items()
                }

                # ----------------- Per-sample logging (testing only, STREAMED) -----------------
                data_dir = os.path.join(data_root, spec.data_dir)
                n_ood = len(ood_conf)

                for fpr_target, tau in tau_map.items():
                    for rp, g_lab, pr, sc in zip(
                        rel_paths, ood_gt_raw, ood_pred, ood_conf
                    ):
                        full_path = os.path.join(data_dir, rp)

                        if spec.role == "id_shift":
                            # ID-like dataset under shift (CNS, ImageNet-C, etc.)
                            tag = spec.parse_tag(rp) if spec.parse_tag is not None else ""
                            scale = (
                                spec.parse_scale(rp)
                                if spec.parse_scale is not None
                                else 0.0
                            )
                            row = build_id_shift_row(
                                dataset_name=spec.name,
                                det_name=det,
                                backbone=backbone,
                                tau=tau,
                                fpr_target=fpr_target,
                                full_path=full_path,
                                shift=tag,
                                scale=scale,
                                gt=int(g_lab),
                                pred=int(pr),
                                score=float(sc),
                            )
                        else:
                            # OOD-like dataset (NINCO, etc.)
                            subset = spec.parse_tag(rp) if spec.parse_tag is not None else ""
                            row = build_ood_row(
                                dataset_name=spec.name,
                                det_name=det,
                                backbone=backbone,
                                tau=tau,
                                fpr_target=fpr_target,
                                full_path=full_path,
                                subset=subset,
                                score=float(sc),
                                pred=int(pr),
                            )

                        # Initialize CSV writer on first row
                        if sample_writer is None:
                            fieldnames = list(row.keys())
                            sample_writer = csv.DictWriter(
                                sample_file, fieldnames=fieldnames
                            )
                            sample_writer.writeheader()
                            sample_header_written = True

                        sample_writer.writerow(row)

                # ----------------- OOD-style metrics (OpenOOD) -----------------
                # For metrics: treat dataset as OOD; labels = -1
                ood_gt = -1 * np.ones_like(ood_gt_raw)

                metrics = compute_ood_metrics(
                    id_pred=id_pred,
                    id_conf=id_conf,
                    id_gt=id_gt,
                    ood_pred=ood_pred,
                    ood_conf=ood_conf,
                    ood_gt=ood_gt,
                )
                metrics_row = build_metrics_row(
                    dataset_name=ds_name,
                    backbone=backbone,
                    detector=det,
                    metrics=metrics,
                    n_id=n_id,
                    n_ood=n_ood,
                    subset="ALL",
                )
                metrics_rows.append(metrics_row)

    # ----------------- Close per-sample file -----------------
    sample_file.close()
    if not sample_header_written:
        print("[WARN] No per-sample rows were written (empty benchmark?)")

    # ----------------- Save metrics and thresholds -----------------
    metrics_csv = os.path.join(out_dir, f"psycho_metrics_{trial}.csv")
    thresholds_json = os.path.join(out_dir, f"psycho_thresholds_{trial}.json")

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(metrics_csv, index=False)

    with open(thresholds_json, "w") as f:
        json.dump(thresholds_log, f, indent=2)

    print(f"\n[Done] Saved per-sample rows → {per_sample_csv}")
    print(f"[Done] Saved metrics rows   → {metrics_csv}")
    print(f"[Done] Saved thresholds     → {thresholds_json}")
    if not df_metrics.empty:
        print("Metrics columns:", ", ".join(df_metrics.columns))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data",
                    help="Root directory (contains images_largescale, benchmark_imglist, etc.)")
    ap.add_argument("--id_name", type=str, default="imagenet")

    ap.add_argument("--out_dir", type=str, default="results/ood_runs/psycho",
                    help="Where to write CSV/JSON outputs")
    ap.add_argument("--backbones", nargs="+",
                    default=[
                        "resnet50",
                        "vit"

                    ],
                    help="Backbones to evaluate (including psycho variants)")
    ap.add_argument("--detectors", nargs="*", default=DEFAULT_DETECTORS,
                    help="OpenOOD postprocessors to use (e.g., msp ebo nci)")
    ap.add_argument("--fprs", type=float, nargs="*", default=DEFAULT_FPRS,
                    help="FPR targets for threshold calibration on ID")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=["cns", "imagenet_c", "imagenet_ln"],
        help="Datasets to evaluate (must be keys in DATASET_REGISTRY)",
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        data_root=args.data_root,
        out_dir=args.out_dir,
        backbones=args.backbones,
        detectors=args.detectors,
        fprs=args.fprs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        dataset_names=args.datasets,
    )