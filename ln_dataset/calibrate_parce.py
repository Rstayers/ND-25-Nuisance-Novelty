# calibrate_parce.py
"""
Calibrate PaRCE (overall) for ImageNet-LN using your existing components:
- Ensemble perception model: ResNet-50 + ViT-B/16 (ImageNet1K pretrained)
- Reconstruction model: ln_dataset.core.autoencoder.ClassifierAwareAE
- Dataset loader: ln_dataset.utils.ImgListDataset

Outputs a torch file containing:
  - means:  [num_classes] per-class reconstruction-loss mean (mu_c)
  - stds:   [num_classes] per-class reconstruction-loss std  (sigma_c)
  - zscore: scalar z selected s.t. average competency ~= accuracy (Eq. 8)
  - meta:   useful run metadata

This calibration file is what your ConfidenceJudge should load to compute the
actual PaRCE overall score (Eq. 7) instead of the current MSP*validity heuristic.

Example:
  python calibrate_parce.py \
    --data /path/to/imagenet \
    --imglist /path/to/val_imglist.txt \
    --ae_weights /path/to/ae_weights.pth \
    --out /path/to/parce_calib_imagenet.pt \
    --batch_size 64 \
    --z_calib_images 10000
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from ln_dataset.core.autoencoder import ClassifierAwareAE
from ln_dataset.utils import ImgListDataset, NORM_MEAN, NORM_STD


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def parce_overall_scores(
    rec_loss: torch.Tensor,          # [B]
    class_probs: torch.Tensor,       # [B, C] already softmaxed
    means: torch.Tensor,             # [C]
    stds: torch.Tensor,              # [C]
    z: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Implements Eq. (7):
      rho_hat(X) = p_hat_c_hat * sum_c p_hat_c * (1 - Phi((ell(X)-2*mu_c)/sigma_c - z))
    where p_hat_c_hat is the MSP.
    """
    # [B]
    msp = class_probs.max(dim=1).values

    # Broadcast to [B, C]
    mu = means[None, :]
    sig = (stds[None, :] + eps)

    zvals = (rec_loss[:, None] - 2.0 * mu) / sig - float(z)     # [B, C]
    loss_probs = 1.0 - normal_cdf(zvals)                        # [B, C]

    scores = torch.sum(loss_probs * class_probs, dim=1)         # [B]
    scores = scores * msp                                       # [B]
    return scores


def _collate_filter_none(batch):
    """
    ImgListDataset sometimes yields img=None; filter those out.
    Expects elements like (img, label, path).
    """
    batch = [b for b in batch if b is not None and b[0] is not None]
    if len(batch) == 0:
        return None
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs, dim=0), torch.tensor(labels, dtype=torch.long), list(paths)


@dataclass
class Reservoir:
    k: int
    seed: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.n_seen = 0
        self.losses: List[float] = []
        self.probs: List[np.ndarray] = []
        self.msps: List[float] = []
        self.correct: List[int] = []

    def consider(self, loss: float, prob_vec: np.ndarray, msp: float, is_correct: bool):
        self.n_seen += 1
        if self.k <= 0:
            return
        if len(self.losses) < self.k:
            self.losses.append(loss)
            self.probs.append(prob_vec)
            self.msps.append(msp)
            self.correct.append(1 if is_correct else 0)
            return

        # Reservoir sampling
        j = self.rng.randint(1, self.n_seen)
        if j <= self.k:
            idx = j - 1
            self.losses[idx] = loss
            self.probs[idx] = prob_vec
            self.msps[idx] = msp
            self.correct[idx] = 1 if is_correct else 0

    def to_tensors(self, device: torch.device, probs_dtype: torch.dtype = torch.float16):
        if len(self.losses) == 0:
            raise ValueError("Reservoir is empty; increase --z_calib_images or check your dataset.")
        losses = torch.tensor(self.losses, dtype=torch.float32, device=device)
        probs = torch.tensor(np.stack(self.probs, axis=0), dtype=probs_dtype, device=device)
        msps = torch.tensor(self.msps, dtype=torch.float32, device=device)
        correct = torch.tensor(self.correct, dtype=torch.float32, device=device)
        return losses, probs, msps, correct


def calibrate_parce(
    data_root: str,
    imglist_path: str,
    ae_weights_path: str,
    out_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    max_images: Optional[int] = None,
    num_classes: int = 1000,
    z_min: float = 0.0,
    z_max: float = 5.0,
    z_step: float = 0.05,
    z_calib_images: int = 10000,
    reservoir_seed: int = 0,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs PaRCE calibration:
      1) estimate per-class mu_c, sigma_c of reconstruction loss using holdout set
      2) select z to nearest 0.05 s.t. mean competency ~= accuracy (Eq. 8)

    Returns the saved dict.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    torch.backends.cudnn.benchmark = (dev.type == "cuda")

    # Models
    print("Loading perception models (ResNet-50, ViT-B/16) and reconstruction AE...")
    resnet = models.resnet50(weights="IMAGENET1K_V1").to(dev).eval()
    vit = models.vit_b_16(weights="IMAGENET1K_V1").to(dev).eval()

    ae = ClassifierAwareAE().to(dev)
    ae.load_state_dict(torch.load(ae_weights_path, map_location=dev))
    ae.eval()

    normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    # Dataset / loader (match generate_ln.py resize/toTensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImgListDataset(data_root, imglist_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=_collate_filter_none,
        drop_last=False,
    )

    # Per-class running stats for reconstruction loss
    counts = np.zeros((num_classes,), dtype=np.int64)
    sums = np.zeros((num_classes,), dtype=np.float64)
    sumsq = np.zeros((num_classes,), dtype=np.float64)

    total = 0
    correct_total = 0

    reservoir = Reservoir(k=z_calib_images, seed=reservoir_seed)

    print("Pass 1: collecting reconstruction-loss stats and sampling for z calibration...")
    pbar = tqdm(loader, total=None)
    for batch in pbar:
        if batch is None:
            continue
        imgs, labels, _paths = batch
        if max_images is not None and total >= max_images:
            break

        # Trim batch if we are about to exceed max_images
        if max_images is not None and total + imgs.size(0) > max_images:
            keep = max_images - total
            imgs = imgs[:keep]
            labels = labels[:keep]

        imgs = imgs.to(dev, non_blocking=True)
        labels = labels.to(dev, non_blocking=True)

        with torch.inference_mode():
            # Perception ensemble probs
            img_norm = normalize(imgs)
            logits_r = resnet(img_norm)
            logits_v = vit(img_norm)
            probs_r = F.softmax(logits_r, dim=1)
            probs_v = F.softmax(logits_v, dim=1)
            probs = (probs_r + probs_v) / 2.0  # [B, C]

            preds = probs.argmax(dim=1)
            msps = probs.max(dim=1).values
            is_correct = (preds == labels)

            # Reconstruction loss (mean MSE)
            ae_out = ae(imgs)
            rec = ae_out[0] if isinstance(ae_out, tuple) else ae_out
            loss_map = F.mse_loss(rec, imgs, reduction="none")
            rec_loss = loss_map.view(loss_map.size(0), -1).mean(dim=1)  # [B]

        # Update aggregate accuracy
        bsz = imgs.size(0)
        total += bsz
        correct_total += int(is_correct.sum().item())
        acc = correct_total / max(total, 1)

        # Update per-class stats (by TRUE label)
        rec_loss_cpu = rec_loss.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        for l, loss_val in zip(labels_cpu, rec_loss_cpu):
            if 0 <= int(l) < num_classes:
                counts[int(l)] += 1
                sums[int(l)] += float(loss_val)
                sumsq[int(l)] += float(loss_val) * float(loss_val)

        # Reservoir sample for z calibration
        probs_cpu = probs.detach().float().cpu().numpy()
        msps_cpu = msps.detach().float().cpu().numpy()
        corr_cpu = is_correct.detach().cpu().numpy()
        for i in range(bsz):
            reservoir.consider(
                loss=float(rec_loss_cpu[i]),
                prob_vec=probs_cpu[i].astype(np.float16, copy=False),
                msp=float(msps_cpu[i]),
                is_correct=bool(corr_cpu[i]),
            )

        pbar.set_description(f"seen={total} acc={acc:.4f} reservoir={min(len(reservoir.losses), reservoir.k)}")

    if total == 0:
        raise RuntimeError("No images processed. Check --data / --imglist and ImgListDataset parsing.")

    # Compute per-class mu, sigma
    means = np.zeros((num_classes,), dtype=np.float32)
    stds = np.zeros((num_classes,), dtype=np.float32)

    global_mean = float(sums.sum() / max(counts.sum(), 1))
    global_var = float((sumsq.sum() / max(counts.sum(), 1)) - global_mean * global_mean)
    global_std = float(math.sqrt(max(global_var, 1e-12)))

    for c in range(num_classes):
        if counts[c] >= 2:
            mu = sums[c] / counts[c]
            var = (sumsq[c] / counts[c]) - mu * mu
            means[c] = float(mu)
            stds[c] = float(math.sqrt(max(var, 1e-12)))
        elif counts[c] == 1:
            means[c] = float(sums[c] / counts[c])
            stds[c] = float(max(global_std, 1e-6))
        else:
            means[c] = float(global_mean)
            stds[c] = float(max(global_std, 1e-6))

    # Z calibration (Eq. 8) on reservoir subset
    print("Pass 2: selecting z-score so mean competency matches accuracy (Eq. 8)...")
    subset_losses, subset_probs, subset_msps, subset_correct = reservoir.to_tensors(
        device=dev, probs_dtype=torch.float16
    )
    subset_probs_f = subset_probs.float()  # do math in fp32 for stability
    subset_acc = float(subset_correct.mean().item())

    means_t = torch.tensor(means, dtype=torch.float32, device=dev)
    stds_t = torch.tensor(stds, dtype=torch.float32, device=dev)

    # Precompute base term: (loss - 2*mu)/sigma
    base = (subset_losses[:, None] - 2.0 * means_t[None, :]) / (stds_t[None, :] + 1e-6)  # [K, C]

    zs = np.round(np.arange(z_min, z_max + 1e-9, z_step), 2).tolist()
    best_z = None
    best_diff = float("inf")
    best_mean_score = None

    # Evaluate in chunks if K is large (avoid GPU OOM)
    # base is already [K,C]; for K=10k, C=1000 -> 10M floats ~ 40MB fp32, OK on most GPUs.
    # If you go larger, reduce --z_calib_images.
    for z in tqdm(zs):
        with torch.inference_mode():
            zvals = base - float(z)                  # [K, C]
            loss_probs = 1.0 - normal_cdf(zvals)     # [K, C]
            scores = torch.sum(loss_probs * subset_probs_f, dim=1) * subset_probs_f.max(dim=1).values
            mean_score = float(scores.mean().item())
            diff = abs(mean_score - subset_acc)

        if diff < best_diff:
            best_diff = diff
            best_z = float(z)
            best_mean_score = mean_score

    assert best_z is not None

    holdout_acc = correct_total / total
    print(f"Holdout accuracy (full pass):  {holdout_acc:.6f} over N={total}")
    print(f"Subset accuracy (z-calib):     {subset_acc:.6f} over K={len(reservoir.losses)}")
    print(f"Selected z:                    {best_z:.2f}")
    print(f"Mean competency at z (subset): {best_mean_score:.6f}")
    print(f"|mean_score - subset_acc|:     {best_diff:.6f}")

    out = {
        "means": torch.tensor(means, dtype=torch.float32),
        "stds": torch.tensor(stds, dtype=torch.float32),
        "zscore": float(best_z),
        "meta": {
            "data_root": os.path.abspath(data_root),
            "imglist": os.path.abspath(imglist_path),
            "ae_weights": os.path.abspath(ae_weights_path),
            "num_classes": int(num_classes),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "max_images": None if max_images is None else int(max_images),
            "z_grid": {"min": float(z_min), "max": float(z_max), "step": float(z_step)},
            "z_calib_images": int(z_calib_images),
            "reservoir_seen": int(reservoir.n_seen),
            "holdout_total": int(total),
            "holdout_acc_full": float(holdout_acc),
            "z_calib_subset_acc": float(subset_acc),
            "selected_z_mean_competency": float(best_mean_score),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved PaRCE calibration to: {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Image root (same as generate_ln.py --data)")
    parser.add_argument("--imglist", type=str, required=True, help="Imglist file (same as generate_ln.py --imglist)")
    parser.add_argument("--ae_weights", type=str, required=True, help="Path to ClassifierAwareAE weights")
    parser.add_argument("--out", type=str, required=True, help="Output .pt file to save calibration")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on processed images (debug)")
    parser.add_argument("--num_classes", type=int, default=1000)

    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=5.0)
    parser.add_argument("--z_step", type=float, default=0.05)
    parser.add_argument("--z_calib_images", type=int, default=10000, help="Reservoir size used to pick z")
    parser.add_argument("--reservoir_seed", type=int, default=0)

    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / cuda:0, etc.")
    args = parser.parse_args()

    calibrate_parce(
        data_root=args.data,
        imglist_path=args.imglist,
        ae_weights_path=args.ae_weights,
        out_path=args.out,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
        num_classes=args.num_classes,
        z_min=args.z_min,
        z_max=args.z_max,
        z_step=args.z_step,
        z_calib_images=args.z_calib_images,
        reservoir_seed=args.reservoir_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
