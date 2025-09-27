import argparse
import os
import sys
import time
import json
import hashlib
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torchvision import models

# OpenOOD unified evaluation API (uses the signature you pasted)
from openood.evaluation_api import Evaluator
from openood.networks.resnet50 import ResNet50
# ---------------------------------------------------------
# Nuisance Novelty Collector (detector sweep)
# ---------------------------------------------------------

DEFAULT_DETECTORS = [
    "vim",
    "msp",
    "energy",


]

DEFAULT_FPRS = [0.01, 0.05, 0.10]

def _require_cuda_or_exit():
    """Evaluator internally calls .cuda(); fail fast if CUDA is unavailable."""
    if not torch.cuda.is_available():
        print("[FATAL] CUDA is not available but OpenOOD Evaluator expects CUDA tensors.\n"
              "        Please run on a GPU-enabled environment or modify OpenOOD to support CPU.",
              file=sys.stderr)
        sys.exit(1)


def _build_net(backbone: str, device: str):
    if backbone.lower() == "resnet50":
        net = ResNet50(num_classes=1000)
        # Load pretrained weights if you have them
        ckpt = torch.load("data/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt", map_location=device)
        net.load_state_dict(ckpt)
        net.eval().to(device)
        class_names = [str(i) for i in range(1000)]  # replace if you have labels
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return net, class_names


def _id_paths_from_evaluator(evaluator: Evaluator) -> List[str]:
    id_ds = evaluator.dataloader_dict['id']['test'].dataset
    return [
        os.path.join(id_ds.data_dir, line.split(' ', 1)[0])
        for line in id_ds.imglist
    ]


def _calibrate_thresholds(id_conf: np.ndarray, fprs: List[float]) -> Dict[float, float]:
    # Larger score = more ID-like for common postprocessors used here
    return {fpr: float(np.quantile(id_conf, fpr)) for fpr in fprs}


def _row(
    trial_id: str,
    dataset: str,
    image_path: str,
    gt: int,
    pred: int,
    score: float,
    detector: str,
    backbone: str,
    tau: float,
    fpr_target: float,
) -> Dict[str, Any]:
    correct = int(pred == gt)
    accept = int(score >= tau)

    if correct and not accept:
        error_type = "nuisance_novelty"
    elif correct and accept:
        error_type = "correct_accept"
    elif (not correct) and (not accept):
        error_type = "misclass_reject"
    else:
        error_type = "misclass_accept"

    return {
        "trial_id": trial_id,
        "dataset": dataset,
        "split": "id-test",
        "image_path": image_path,
        "class_id": int(gt),
        "pred_class": int(pred),
        "correct_cls": correct,
        "cls_confidence": float(score),  # for MSP/Energy this equals the OOD "ID score"
        "detector": detector,
        "backbone": backbone,
        "score": float(score),
        "threshold": float(tau),
        "fpr_target": float(fpr_target),
        "accept": accept,
        "is_nn": int(error_type == "nuisance_novelty"),
        "error_type": error_type,
        # Placeholders for nuisance tags (can be filled if using IN-C/ObjectNet, etc.)
        "corruption_type": None,
        "severity": None,
        "pose_bin": None,
        "bg_type": None,
        "occlusion_level": None,
        # Repro/params
        "postprocess_config": detector,
    }


def run(
    data_root: str,
    out_dir: str,
    backbone: str,
    detectors: List[str],
    fprs: List[float],
    batch_size: int,
    num_workers: int,
    seed: int,
):
    _require_cuda_or_exit()
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    device = "cuda"
    net, class_names = _build_net(backbone, device)

    trial_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    all_rows: List[Dict[str, Any]] = []
    thresholds_log: Dict[str, Dict[str, float]] = {}

    for detector in detectors:
        print(f"\n[Init Evaluator] id=imagenet, postprocessor={detector}")


        evaluator = Evaluator(
            net,
            id_name="imagenet",
            data_root=data_root,
            config_root='configs',          # let OpenOOD resolve default configs
            preprocessor=None,         # default preprocessor for ImageNet
            postprocessor_name=detector,
            postprocessor=None,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Run OOD eval once to populate scores (and compute standard metrics)
        evaluator.eval_ood(fsood=False, progress=True)

        # Pull ID test predictions/confidences/labels gathered by Evaluator
        id_pred, id_conf, id_gt = evaluator.scores['id']['test']
        id_paths = _id_paths_from_evaluator(evaluator)

        # Calibrate thresholds on ID-test distribution as a proxy for ID-val
        # (If you have a true ID-val split exposed by OpenOOD, swap in here.)
        tau_map = _calibrate_thresholds(id_conf, fprs)
        thresholds_log[detector] = {str(k): float(v) for k, v in tau_map.items()}

        # Emit one row per image per FPR target
        for fpr, tau in tau_map.items():
            for path, gt, pred, score in zip(id_paths, id_gt, id_pred, id_conf):
                all_rows.append(
                    _row(
                        trial_id=trial_id,
                        dataset="imagenet",
                        image_path=path,
                        gt=int(gt),
                        pred=int(pred),
                        score=float(score),
                        detector=detector,
                        backbone=backbone,
                        tau=float(tau),
                        fpr_target=float(fpr),
                    )
                )

    # Save unified CSV + thresholds JSON
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, f"nuisance_novelty_trials_{trial_id}.csv")
    df.to_csv(csv_path, index=False)

    with open(os.path.join(out_dir, f"thresholds_{trial_id}.json"), "w") as f:
        json.dump(thresholds_log, f, indent=2)

    print(f"\n[Done] Saved {len(df)} rows → {csv_path}")
    print("Columns:", ", ".join(df.columns))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data",
                    help="Root containing benchmark_imglist and images_* dirs")
    ap.add_argument("--out_dir", type=str, default="results",
                    help="Where to write CSV/JSON outputs")
    ap.add_argument("--backbone", type=str, default="resnet50",
                    help="Classifier backbone (supported: resnet50)")
    ap.add_argument("--detectors", type=str, nargs="*", default=DEFAULT_DETECTORS,
                    help="OpenOOD postprocessor names to sweep")
    ap.add_argument("--fprs", type=float, nargs="*", default=DEFAULT_FPRS,
                    help="Target ID-FPRs for thresholding")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_root=args.data_root.rstrip("/"),
        out_dir=args.out_dir,
        backbone=args.backbone,
        detectors=args.detectors,
        fprs=args.fprs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
