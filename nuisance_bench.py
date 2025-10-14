import argparse, os, sys, time, json, hashlib
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from openood.evaluation_api import Evaluator
from openood.networks.resnet50 import ResNet50
from torchvision.models import vit_b_16, ViT_B_16_Weights
from openood.networks.vit_b_16 import ViT_B_16

# -----------------------------
# Config
# -----------------------------
DEFAULT_DETECTORS = ["vim",
                     "knn",
                     "she"]
DEFAULT_FPRS = [0.01, 0.05, 0.1]


def _require_cuda_or_exit():
    if not torch.cuda.is_available():
        print("[FATAL] CUDA is required by OpenOOD Evaluator. Exiting.",
              file=sys.stderr)
        sys.exit(1)


def _trial_id() -> str:
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:8]


def _calibrate_thresholds(id_conf: np.ndarray, fprs: List[float]) -> Dict[float, float]:
    """Calibrate thresholds so that ID-FPR ≈ target."""
    return {float(f): float(np.quantile(id_conf, f)) for f in fprs}



def parse_from_path(path: str):
    """
    Parse corruption type and severity from an ImageNet-C filepath.
    Example: .../imagenet_c/gaussian_noise/3/xxx.jpeg → ("gaussian_noise", 3)
    """
    parts = Path(path).parts
    try:
        idx = [i for i, p in enumerate(parts) if p.lower().startswith("imagenet_c")][0]
        corruption = parts[idx+1]
        severity = int(parts[idx+2])
        return corruption, severity
    except Exception:
        return "unknown", -1



def _row(
    dataset: str,
    corruption: str,
    severity: int,
    det_name: str,
    backbone: str,
    tau: float,
    fpr_target: float,
    path: str,
    gt: int,
    pred: int,
    score: float,
) -> Dict[str, Any]:
    correct = int(pred == gt)
    accept = int(score >= tau)

    if correct and not accept:
        err = "Full_Nuisance"
    elif correct and accept:
        err = "Full_Correct"
    elif (not correct) and (not accept):
        err = "Partial_Nuisance"
    else:
        err = "Partial_Correct"

    return {
        "dataset": dataset,
        "corruption": corruption,
        "severity": severity,
        "detector": det_name,
        "backbone": backbone,
        "fpr_target": fpr_target,
        "threshold": tau,
        "image_path": path,
        "class_id": gt,
        "pred_class": pred,
        "score": score,
        "correct_cls": correct,
        "accept": accept,
        "error_type": err,
        "is_nn": int(err == "nuisance_novelty"),
    }


def _extract_paths(dataloader):
    dset = dataloader.dataset
    if hasattr(dset, "imglist"):  # OpenOOD ImglistDataset
        base = getattr(dset, "data_dir", ".")
        return [os.path.join(base, line.split(" ", 1)[0]) for line in dset.imglist]
    elif hasattr(dset, "samples"):  # torchvision datasets
        return [s[0] for s in dset.samples]
    else:
        return [""] * len(dset)


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

    # Load backbone
    if backbone.lower() == "resnet50":
        net = ResNet50(num_classes=1000)
        ckpt_path = "data/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt"
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt)
        net.eval().to(device)
    elif backbone.lower() in ["vit", "vit_b_16"]:
        tv_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        net = ViT_B_16(num_classes=1000)
        state_dict = tv_model.state_dict()

        # Load into your wrapper
        net.load_state_dict(state_dict, strict=False)
        net.eval().to(device)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")



    trial_id = _trial_id()
    all_rows = []
    thresholds_log: Dict[str, Dict[str, float]] = {}

    for det in detectors:
        print(f"\n[Init Evaluator] id=imagenet, postprocessor={det}")
        evaluator = Evaluator(
            net=net,
            id_name="imagenet",
            data_root=data_root,
            config_root="configs",
            preprocessor=None,
            postprocessor_name=det,
            postprocessor=None,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # -------------------------
        # ID inference
        # -------------------------
        print("Running ID test inference...")
        id_pred, id_conf, id_gt = evaluator.postprocessor.inference(
            evaluator.net, evaluator.dataloader_dict["id"]["test"], progress=True
        )
        evaluator.scores["id"]["test"] = [id_pred, id_conf, id_gt]
        id_paths = _extract_paths(evaluator.dataloader_dict["id"]["test"])

        tau_map = _calibrate_thresholds(id_conf, fprs)
        thresholds_log[det] = {str(k): float(v) for k, v in tau_map.items()}

        # -------------------------
        # ImageNet-C only
        # -------------------------
        csid_dict = evaluator.dataloader_dict.get("csid", {})

        imagenet_c_splits = {k: v for k, v in csid_dict.items() if "imagenet_c" in k}
        if not imagenet_c_splits:
            raise RuntimeError("[ERROR] No ImageNet-C splits found!")

        for csid_name, dataloader in imagenet_c_splits.items():

            pred, conf, gt = evaluator.postprocessor.inference(
                evaluator.net, dataloader, progress=True
            )
            paths = _extract_paths(dataloader)

            for fpr, tau in tau_map.items():
                for p, g, pr, sc in zip(paths, gt, pred, conf):
                    corruption, severity = parse_from_path(p)
                    all_rows.append(
                        _row(
                            dataset="imagenet_c",
                            corruption=corruption,
                            severity=severity,
                            det_name=det,
                            backbone=backbone,
                            tau=tau,
                            fpr_target=fpr,
                            path=p,
                            gt=int(g),
                            pred=int(pr),
                            score=float(sc),
                        )
                    )

    # --- Save outputs
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, f"nuisance_imagenet_c_{trial_id}.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, f"thresholds_{trial_id}.json"), "w") as f:
        json.dump(thresholds_log, f, indent=2)

    print(f"\n[Done] Saved {len(df)} rows → {csv_path}")
    print("Columns:", ", ".join(df.columns))

    # --- Summaries
    print("\n[Summary: Nuisance rate by severity]")
    sev_summary = (
        df.groupby(["detector", "fpr_target", "severity"])
        .agg(nn_rate=("is_nn", "mean"), n=("is_nn", "size"))
        .reset_index()
    )
    print(sev_summary.to_string(index=False))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data",
                    help="Root containing benchmark_imglist and images_* dirs")
    ap.add_argument("--out_dir", type=str, default="results/nuisance_runs",
                    help="Where to write CSV/JSON outputs")
    ap.add_argument("--backbone", type=str, default="vit")
    ap.add_argument("--detectors", type=str, nargs="*", default=DEFAULT_DETECTORS)
    ap.add_argument("--fprs", type=float, nargs="*", default=DEFAULT_FPRS)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
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
