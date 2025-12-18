import json

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import os
import argparse
import numpy as np
from tqdm import tqdm

# Core Logic
from ln_dataset.core.autoencoder import ClassifierAwareAE
from ln_dataset.core.masks import generate_competency_mask_hybrid

# Nuisance Modules
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance
from ln_dataset.utils import ImgListDataset, STATS_FILE
from ln_dataset.utils import NORM_MEAN, NORM_STD, save_debug_maps, save_tensor_as_img
import math
import torch


# ==========================================
# 1. CONFIGURATION
# ==========================================

BIN_EDGES = {
    '1': 0.930,  # > 0.930 = Level 1
    '2': 0.908,  # > 0.908 = Level 2
    '3': 0.867,  # > 0.867 = Level 3
    '4': 0.749,  # > 0.749 = Level 4
    '5': 0.000   # <= 0.749 = Level 5
}
DEBUG_MAX_IMAGES = 10

def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
# ==========================================
# 2. CONFIDENCE (PaRCE) JUDGE
# ==========================================
class ConfidenceJudge:
    """
    Implements PaRCE scoring (Probability + Reconstruction Competency Estimation).
    Renamed to 'ConfidenceJudge' to maintain compatibility with existing scripts.
    """

    def __init__(self, device, ae_weights_path, parce_calib_path=STATS_FILE):
        self.device = device

        print("Loading classifiers and autoencoder...")
        # 1. Load Classifiers (Ensemble)
        self.resnet = models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()

        # 2. Load Autoencoder (The "Reconstructive" part of PaRCE)
        self.ae = ClassifierAwareAE().to(device)
        self.ae.load_state_dict(torch.load(ae_weights_path, map_location=device))
        self.ae.eval()

        # Normalizer for classifiers
        self.normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        calib = torch.load(parce_calib_path, map_location="cpu")
        self.parce_means = calib["means"].to(device).float()   # [1000]
        self.parce_stds  = calib["stds"].to(device).float()    # [1000]
        self.parce_z     = float(calib["zscore"])
        self.parce_eps   = 1e-6

    @torch.no_grad()
    def get_competency(self, img_tensor, target_label=None):
        # A) classifier probs (your ensemble)
        img_norm = self.normalize(img_tensor)
        probs_r = torch.softmax(self.resnet(img_norm), dim=1)
        probs_v = torch.softmax(self.vit(img_norm), dim=1)
        class_probs = (probs_r + probs_v) / 2.0  # [B,1000]

        msp, pred_tensor = torch.max(class_probs, dim=1)  # [B]
        pred_idx = int(pred_tensor.item())

        # B) reconstruction loss ℓ(X) (mean MSE)
        rec_img = self.ae(img_tensor)
        if isinstance(rec_img, tuple):
            rec_img = rec_img[0]
        loss = torch.mean((rec_img - img_tensor) ** 2, dim=(1, 2, 3))  # [B]

        # C) actual PaRCE overall score (Eq. 7)
        means = self.parce_means[None, :]  # [1,1000]
        stds = (self.parce_stds[None, :] + self.parce_eps)
        zvals = (loss[:, None] - 2.0 * means) / stds - self.parce_z  # [B,1000]
        loss_probs = 1.0 - normal_cdf(zvals)  # [B,1000]

        scores = torch.sum(loss_probs * class_probs, dim=1)  # [B]
        scores = scores * msp  # multiply by p_hat_c_hat (overall)
        parce_score = float(scores.item())

        disagree = (probs_r.argmax(dim=1).item() != probs_v.argmax(dim=1).item())

        if target_label is not None:
            debug_str = f"PaRCE:{parce_score:.3f} MSP:{float(msp.item()):.3f} loss:{float(loss.item()):.4f} z:{self.parce_z:.2f}"
        else:
            debug_str = "N/A"

        return parce_score, pred_idx, disagree, debug_str


def sweep_and_select(judge, ae_model, img, label, output_dir, base_name, debug_dir=None):
    # 1. Generate Mask
    mask = generate_competency_mask_hybrid(ae_model, img, models=[judge.resnet, judge.vit], area=0.10,
                                           avoid_top_saliency=0.4, contiguous=False)

    if debug_dir is not None:
        save_debug_maps(judge, ae_model, img, label, mask, debug_dir)

    # 2. Define Nuisances (Standard + Bidirectional Photometric)
    # We include tuples of (NuisanceObject, FolderName)
    nuisances = [
        (LocalNoiseNuisance(severity=1), 'noise'),
        (LocalPixelationNuisance(severity=1), 'pixelation'),
       (LocalSpatialNuisance(severity=1), 'spatial'),
        (LocalPhotometricNuisance(mode='brightness', severity=1), 'brightness'),
        (LocalPhotometricNuisance(mode='contrast', severity=1), 'contrast'),
        (LocalPhotometricNuisance(mode='saturation', severity=1), 'saturation'),
    ]

    selected_samples = []

    for nuisance_obj, n_name in nuisances:

        # Dictionary to store the BEST (hardest) candidate for each level found so far
        # Key: Level ('1', '2'...), Value: dict with image, p, score
        best_candidates = {}

        # Sweep severity p (Forward: 0.05 -> 1.0)
        # We MUST go forward to detect where the model breaks
        for p in np.linspace(0.05, 1.0, 50):

            # A. Apply Nuisance
            img_perturbed = nuisance_obj.apply(img, mask, manual_param=p)

            # B. Measure Competency
            parce, pred, disagree, _ = judge.get_competency(
                img_perturbed,
                target_label=label,
            )

            # C. STRICT ENFORCEMENT
            # If model fails, we cannot go deeper. STOP scanning this nuisance.
            if disagree or (pred != label):
                break

            # D. Identify Bin
            assigned_lvl = None
            if parce > BIN_EDGES['1']:
                assigned_lvl = '1'
            elif parce > BIN_EDGES['2']:
                assigned_lvl = '2'
            elif parce > BIN_EDGES['3']:
                assigned_lvl = '3'
            elif parce > BIN_EDGES['4']:
                assigned_lvl = '4'
            else:
                assigned_lvl = '5'

            # E. UPDATE CANDIDATE (The Fix)
            # Since p is increasing, every new match for a level is "harder" (higher p)
            # We simply overwrite the previous candidate for this level.
            if assigned_lvl:
                best_candidates[assigned_lvl] = {
                    'img': img_perturbed,  # Keep tensor on GPU until save
                    'p': p,
                    'score': parce,
                    'level': assigned_lvl
                }

        # F. SAVE RESULTS
        # Now that the loop is done (or broken), we save the hardest candidates we found.
        for lvl, data in best_candidates.items():
            level_dir = os.path.join(output_dir, n_name, lvl)
            os.makedirs(level_dir, exist_ok=True)

            fname = f"{base_name}.png"
            save_path = os.path.join(level_dir, fname)

            save_tensor_as_img(data['img'], save_path)

            rel_path = os.path.join(n_name, lvl, fname).replace("\\", "/")
            selected_samples.append({'path': rel_path})

    return selected_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--imglist', type=str, required=True)
    parser.add_argument('--ae_weights', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="./ln_output_v3")
    parser.add_argument('--debug_maps', action='store_true')
    parser.add_argument('--debug_max', type=int, default=DEBUG_MAX_IMAGES)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = ClassifierAwareAE().to(device)
    ae.load_state_dict(torch.load(args.ae_weights, map_location=device))
    ae.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    judge = ConfidenceJudge(device, ae_weights_path=args.ae_weights)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_lines = []

    print(f"Starting Generation (Full PaRCE)...")


    for i in tqdm(range(len(dataset))):
        img, label, path = dataset[i]
        if img is None: continue

        img = img.unsqueeze(0).to(device)
        base_name = os.path.splitext(os.path.basename(path))[0]

        parce, pred, disagree, debug_info = judge.get_competency(img, target_label=label)

        if i < DEBUG_MAX_IMAGES:
            print(f"\n[DEBUG Img {i}] L:{label} P:{pred} Dis:{disagree} PaRCE:{parce:.3f} | {debug_info}")

        if pred != label:
            continue
        if disagree:
            continue

        debug_dir = None
        if args.debug_maps and i < args.debug_max:
            debug_dir = os.path.join(args.out_dir, "debug", f"{i:05d}_{base_name}")
        results = sweep_and_select(judge, ae, img, label, args.out_dir, base_name, debug_dir)
        for res in results:
            manifest_lines.append(f"{res['path']} {label}")

    with open(os.path.join(args.out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest_lines))

    print("\nGeneration Complete.")


if __name__ == "__main__":
    main()