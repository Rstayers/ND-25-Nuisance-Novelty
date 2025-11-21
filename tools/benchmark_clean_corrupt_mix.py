#!/usr/bin/env python3
"""
Benchmark OOD detection for fine-tuned ImageNet-C Psycho models and baseline.

Each fine-tuned model:
 - Calibrates on ImageNet ID test set (via OpenOOD Evaluator)
 - Tests OOD ONLY on its matching imagenet_c_psycho split
   e.g. 50clean_50c_resnet50_openood → 50clean_50c
Baseline:
 - Uses pretrained ResNet50 checkpoint
 - Tests on ALL imagenet_c_psycho splits

Outputs aggregated results:
  results/imagenet_c_psycho_benchmark.csv
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from openood.evaluation_api import Evaluator
from openood.networks.resnet50 import ResNet50

# -----------------------------
# Config
# -----------------------------
DEFAULT_DETECTOR = "msp"
FPR_TARGET = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Baseline checkpoint (pretrained, not fine-tuned)
BASELINE_PATH = "../data/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt"

# Psycho imglist base (we will scan subfolders)
OOD_BASE_DIR_REL = "benchmark_imglist/imagenet_c_psycho"


# -----------------------------
# Helpers
# -----------------------------
def _require_cuda():
    if not torch.cuda.is_available():
        print("[FATAL] CUDA required for OpenOOD Evaluator.", file=sys.stderr)
        sys.exit(1)


def _calibrate_threshold(id_conf: np.ndarray, fpr: float) -> float:
    """Return threshold tau such that ID-FPR ≈ target."""
    return float(np.quantile(id_conf, fpr))


def _load_net(ckpt_path: str) -> torch.nn.Module:
    """Load ResNet50 from ckpt. Handles both plain state_dict and {'net': state_dict}."""
    net = ResNet50(num_classes=1000)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "net" in ckpt:
        state = ckpt["net"]
    else:
        state = ckpt
    net.load_state_dict(state, strict=False)
    return net.to(DEVICE).eval()


class PsychoImglistDataset(Dataset):
    """Reads OpenOOD-style imglist and returns (tensor, label)."""

    def __init__(self, base_dir: str, imglist_path: str, transform):
        self.base_dir = base_dir
        self.transform = transform
        self.samples = []

        with open(imglist_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path, label = parts[0], parts[1]
                full_path = os.path.join(self.base_dir, rel_path)
                self.samples.append((full_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def _default_imagenet_transform():
    # Fallback if we can't grab transform from OpenOOD dataset
    from torchvision import transforms
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _matched_split_for_model(model_name: str):
    """
    Extract split name like '50clean_50c' from model dir name
    e.g. '50clean_50c_resnet50_openood' -> '50clean_50c'
    Returns None for baseline, which we treat specially.
    """
    if model_name == "baseline":
        return None
    m = re.search(r"(\d+clean_\d+c)", model_name)
    if m:
        return m.group(1)
    return None


# -----------------------------
# Core evaluation
# -----------------------------
def evaluate_model(model_name: str, ckpt_path: str, data_root: str) -> List[Dict[str, Any]]:
    """Evaluate one model: ID calibration + OOD tests."""
    print(f"\n[Model: {model_name}] Loading checkpoint: {ckpt_path}")
    net = _load_net(ckpt_path)

    # Build standard evaluator for ImageNet using existing OpenOOD configs
    evaluator = Evaluator(
        net=net,
        id_name="imagenet",
        data_root=data_root,
        config_root="configs",
        preprocessor=None,
        postprocessor_name=DEFAULT_DETECTOR,
        postprocessor=None,
        batch_size=256,
        shuffle=False,
        num_workers=8,
    )

    # -------------------------
    # ID inference / calibration via OpenOOD
    # -------------------------
    print("→ Running ID inference (ImageNet)...")
    id_loader = evaluator.dataloader_dict["id"]["test"]
    id_pred, id_conf, id_gt = evaluator.postprocessor.inference(
        evaluator.net, id_loader, progress=False
    )
    id_pred = np.array(id_pred)
    id_conf = np.array(id_conf)
    id_gt = np.array(id_gt)

    id_acc = float((id_pred == id_gt).mean()) * 100.0
    tau = _calibrate_threshold(id_conf, FPR_TARGET)
    print(f"   ID acc: {id_acc:.2f}% | Calibrated τ={tau:.4f}")

    # Grab base_dir and transform from the ID dataset so we match OpenOOD preprocessing
    id_dataset = id_loader.dataset
    id_base_dir = getattr(id_dataset, "data_dir", data_root)
    # Try a couple of likely attributes for the transform
    ood_transform = getattr(id_dataset, "preprocessor", None)
    if ood_transform is None and hasattr(id_dataset, "transform_image"):
        ood_transform = getattr(id_dataset, "transform_image")
    if ood_transform is None and hasattr(id_dataset, "transform"):
        ood_transform = getattr(id_dataset, "transform")
    if ood_transform is None:
        ood_transform = _default_imagenet_transform()

    results: List[Dict[str, Any]] = []

    # -------------------------
    # Decide which psycho splits this model should be tested on
    # -------------------------
    ood_root = Path(data_root) / OOD_BASE_DIR_REL
    matched_split = _matched_split_for_model(model_name)

    split_files: List[Path]
    if matched_split is None:
        # baseline: evaluate on ALL splits
        split_files = sorted(ood_root.glob("*/test_imagenet_c_psycho.txt"))
    else:
        # fine-tuned: only on its matching split
        split_file = ood_root / matched_split / "test_imagenet_c_psycho.txt"
        if not split_file.exists():
            print(f"[WARN] No psycho split file for {model_name} at {split_file}, skipping OOD.")
            return results
        split_files = [split_file]

    if not split_files:
        print(f"[WARN] No OOD splits found under {ood_root}")
        return results

    softmax = torch.nn.Softmax(dim=1)

    for split_file in split_files:
        split = split_file.parent.name
        print(f"→ Evaluating OOD split: {split}")

        ood_dataset = PsychoImglistDataset(
            base_dir=id_base_dir,
            imglist_path=str(split_file),
            transform=ood_transform,
        )
        ood_loader = DataLoader(
            ood_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        all_pred = []
        all_conf = []
        all_gt = []

        with torch.no_grad():
            for imgs, labels in ood_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels_np = labels.numpy()  # keep labels on CPU as numpy
                logits = net(imgs)
                probs = softmax(logits)
                conf, pred = torch.max(probs, dim=1)

                all_pred.append(pred.cpu().numpy())
                all_conf.append(conf.cpu().numpy())
                all_gt.append(labels_np)

        ood_pred = np.concatenate(all_pred, axis=0)
        ood_conf = np.concatenate(all_conf, axis=0)
        ood_gt = np.concatenate(all_gt, axis=0)

        correct = (ood_pred == ood_gt).astype(int)
        accept = (ood_conf >= tau).astype(int)
        nuisance = (correct == 0) & (accept == 0)

        ood_acc = float(correct.mean()) * 100.0
        nn_rate = float(nuisance.mean())
        delta = ood_acc - id_acc

        results.append(
            {
                "model": model_name,
                "split": split,
                "fpr": FPR_TARGET,
                "id_acc": id_acc,
                "ood_acc": ood_acc,
                "delta": delta,
                "nuisance_rate": nn_rate,
            }
        )

        print(
            f"   OOD acc: {ood_acc:.2f}% | Δ={delta:.2f}% | nn_rate={nn_rate:.3f}"
        )

    return results


# -----------------------------
# Main benchmark
# -----------------------------
def main(data_root: str = "data", results_root: str = "results"):
    _require_cuda()
    os.makedirs(results_root, exist_ok=True)

    # Fine-tuned models: results/*clean_*c_resnet50_openood/checkpoints/final.pth
    tuned_models = sorted(
        Path(results_root).glob("*clean_*c_resnet50_openood/checkpoints/final.pth")
    )

    # Baseline + fine-tuned
    all_models = [("baseline", BASELINE_PATH)] + [
        (m.parent.parent.name, str(m)) for m in tuned_models
    ]

    all_results: List[Dict[str, Any]] = []
    for model_name, ckpt_path in all_models:
        if not Path(ckpt_path).exists():
            print(f"[WARN] Missing checkpoint: {ckpt_path}, skipping {model_name}.")
            continue
        res = evaluate_model(model_name, ckpt_path, data_root)
        all_results.extend(res)

    if not all_results:
        print("[WARN] No results collected.")
        return

    df = pd.DataFrame(all_results)
    out_csv = Path(results_root) / "imagenet_c_psycho_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[Done] Saved summary → {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
