import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torchvision import models, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

# Core Logic
from ln_dataset.core.autoencoder import ClassifierAwareAE, get_reconstruction_error
from ln_dataset.core.masks import generate_competency_mask

# Nuisance Modules
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance

# ==========================================
# 1. CONFIGURATION
# ==========================================
TARGET_LEVELS = {
    '1': 0.95,  # Canonical ( < 0.3 sigma )
    '2': 0.80,  # Good ( ~0.6 sigma )
    '3': 0.55,  # Ambiguous ( ~1.2 sigma )
    '4': 0.25,  # Poor ( ~1.7 sigma )
    '5': 0.05   # OOD ( > 2.5 sigma )
}

# Increased tolerance because the Gaussian curve is steep!
# It's hard to hit exactly 0.55, so 0.55 +/- 0.08 covers [0.47 - 0.63]
TOLERANCE = 0.08
STATS_FILE = "ln_dataset/assets/parce_stats.pt"
MIN_SIGMA_RATIO = 0.10  # Sigma must be at least 10% of Mean to prevent score collapse


# ==========================================
# 2. ACTUAL PARCE JUDGE (Robust Implementation)
# ==========================================
class ActualParceJudge:
    def __init__(self, device, ae_model, stats_path=STATS_FILE):
        self.device = device
        self.ae = ae_model

        print("Loading classifiers...")
        self.resnet = models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()

        if os.path.exists(stats_path):
            self.stats = torch.load(stats_path, map_location=device)
            print(f"Loaded PaRCE stats for {len(self.stats)} classes.")
        else:
            print(f"WARNING: No stats found at {stats_path}! Run with --calibrate first.")
            self.stats = {}

    def get_competency(self, img_tensor, target_label=None):
        """
        Calculates PaRCE Score using Robust Half-Gaussian P(ID).
        """
        with torch.no_grad():
            # 1. Individual Forward Passes
            logits_r = self.resnet(img_tensor)
            logits_v = self.vit(img_tensor)

            probs_r = F.softmax(logits_r, dim=1)
            probs_v = F.softmax(logits_v, dim=1)

            # 2. Check Agreement
            pred_r = probs_r.argmax(dim=1).item()
            pred_v = probs_v.argmax(dim=1).item()
            disagreement = (pred_r != pred_v)

            # 3. Ensemble Probability
            avg_probs = (probs_r + probs_v) / 2.0

            if target_label is not None:
                p_class_val = avg_probs[0, target_label].item()
                label_to_use = target_label
                pred_label = avg_probs.argmax(dim=1).item()
            else:
                p_class_val, pred_idx = torch.max(avg_probs, dim=1)
                p_class_val = p_class_val.item()
                label_to_use = pred_idx.item()
                pred_label = label_to_use

            # 4. Reconstruction Probability
            recon = self.ae(img_tensor)
            recon_err = torch.mean((img_tensor - recon) ** 2).item()

            p_id, debug_str = self._calculate_p_id(label_to_use, recon_err)

            # Final Score
            parce_score = p_class_val * p_id

            return parce_score, pred_label, disagreement, debug_str

    def _calculate_p_id(self, label, error):
        """
        Calculates Relative Likelihood of ID.
        Uses One-Sided Gaussian:
        - If error <= mean, we assume it's "Perfectly ID" (Score 1.0)
        - If error > mean, we decay based on Z-score.
        """
        if label not in self.stats:
            return 0.5, "NoStats"

        mu = self.stats[label]['mean']
        sigma = self.stats[label]['std']

        # --- ROBUSTNESS FIX ---
        # If sigma is too small, Z-scores explode and P(ID) -> 0 for everything.
        # We clamp sigma to be at least X% of the mean.
        eff_sigma = max(sigma, mu * MIN_SIGMA_RATIO)

        if error <= mu:
            p_val = 1.0
        else:
            z = (error - mu) / (eff_sigma + 1e-9)
            p_val = np.exp(-0.5 * (z ** 2))

        debug_str = f"Err:{error:.5f} Mu:{mu:.5f} Sig:{eff_sigma:.5f} Z:{z if error > mu else 0:.1f} P_ID:{p_val:.3f}"
        return float(p_val), debug_str


# ==========================================
# 3. CALIBRATION & SWEEP
# ==========================================
def calibrate_parce(ae_model, dataset, device, batch_size=32):
    print("Starting PaRCE Calibration...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    raw_errors = {}
    ae_model.eval()

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader):
            imgs = imgs.to(device)
            recon = ae_model(imgs)
            mse = ((imgs - recon) ** 2).mean(dim=(1, 2, 3))

            for i in range(len(labels)):
                lbl = int(labels[i])
                err = mse[i].item()
                if lbl not in raw_errors: raw_errors[lbl] = []
                raw_errors[lbl].append(err)

    stats = {}
    print("Aggregating Statistics...")
    for lbl, errors in raw_errors.items():
        arr = np.array(errors)
        # Compute Stats
        mu = float(np.mean(arr))
        std = float(np.std(arr))
        stats[lbl] = {'mean': mu, 'std': std}

    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    torch.save(stats, STATS_FILE)
    print(f"Calibration Saved: {STATS_FILE}")
    print(f"Example (Class {list(stats.keys())[0]}): Mu={stats[list(stats.keys())[0]]['mean']:.5f}")


class ImgListDataset(Dataset):
    def __init__(self, root, imglist_path, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        if not os.path.exists(imglist_path): raise FileNotFoundError(f"{imglist_path} not found")
        with open(imglist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2: self.samples.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.root, path)
        try:
            img = Image.open(full_path).convert('RGB')
        except:
            return None, None, None
        if self.transform: img = self.transform(img)
        return img, label, path


def save_tensor_as_img(tensor, path):
    img = torch.clamp(tensor, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def sweep_and_select(judge, ae_model, img, label, output_dir, base_name):
    mask = generate_competency_mask(ae_model, img, percentile=0.40)

    nuisances = [
        LocalNoiseNuisance(severity=1),
        LocalPixelationNuisance(severity=1),
        LocalSpatialNuisance(severity=1),
        LocalPhotometricNuisance(mode='brightness', severity=1),
        LocalPhotometricNuisance(mode='contrast', severity=1),
        LocalPhotometricNuisance(mode='saturation', severity=1),

    ]
    n_names = [ "noise", "pixel", "spatial", "brightness", "contrast", "saturation"]

    selected_samples = []

    for i, nuisance in enumerate(nuisances):
        buckets_needed = set(TARGET_LEVELS.keys())

        # Sweep
        for p in np.linspace(0.05, 1.0, 40):
            if not buckets_needed: break

            img_perturbed = nuisance.apply(img, mask, manual_param=p)

            # PERMISSIVE CHECK for sweep
            parce, pred, disagree, _ = judge.get_competency(img_perturbed, target_label=label)

            for level, target in TARGET_LEVELS.items():
                if level in buckets_needed:
                    if abs(parce - target) < TOLERANCE and (pred == label) and not disagree:
                        # Save
                        n_name = n_names[i]
                        level_dir = os.path.join(output_dir, n_name, str(level))
                        os.makedirs(level_dir, exist_ok=True)

                        fname = f"{base_name}.png"
                        save_path = os.path.join(level_dir, fname)
                        save_tensor_as_img(img_perturbed, save_path)

                        rel_path = os.path.join(n_name, str(level), fname).replace("\\", "/")
                        selected_samples.append({'path': rel_path})

                        buckets_needed.remove(level)
                        break
    return selected_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--imglist', type=str, required=True)
    parser.add_argument('--ae_weights', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="./ln_output_v3")
    parser.add_argument('--calibrate', default=False, action='store_true')
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

    if args.calibrate:
        calibrate_parce(ae, dataset, device)
        exit(0)
    else:
        judge = ActualParceJudge(device, ae_model=ae)
        os.makedirs(args.out_dir, exist_ok=True)
        manifest_lines = []

        print(f"Starting Generation (Strict Base -> Permissive Sweep)...")

        # Debug Counters

        for i in tqdm(range(len(dataset))):
            img, label, path = dataset[i]
            if img is None: continue

            img = img.unsqueeze(0).to(device)
            base_name = os.path.splitext(os.path.basename(path))[0]

            # 1. EVALUATE BASE IMAGE
            parce, pred, disagree, debug_info = judge.get_competency(img, target_label=label)

            # === DEBUG PRINT for first 10 images ===




            if (pred != label) or disagree:
                continue
            # Accepted Base Image
            results = sweep_and_select(judge, ae, img, label, args.out_dir, base_name)
            for res in results:
                manifest_lines.append(f"{res['path']} {label}")

        with open(os.path.join(args.out_dir, "manifest.txt"), "w") as f:
            f.write("\n".join(manifest_lines))

        print("\nGeneration Complete.")
        print(f"Total Processed: {stats['total']}")
        print(f"Skipped (Wrong Pred): {stats['skipped_wrong']}")
        print(f"Skipped (Disagree): {stats['skipped_disagree']}")
        print(f"Skipped (Low Score <0.9): {stats['skipped_low_score']}")
        print(f"Accepted Base Images: {stats['accepted']}")


if __name__ == "__main__":
    main()