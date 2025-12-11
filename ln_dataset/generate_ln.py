import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
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
from ln_dataset.nuisances.blur import LocalBlurNuisance

# --- CONFIGURATION ---
TARGET_LEVELS = {
    '1': 0.95,  # Near perfect
    '2': 0.80,  # Good but uncertain
    '3': 0.60,  # Ambiguous
    '4': 0.40,  # Poor
    '5': 0.15  # Very Low Competency
}
TOLERANCE = 0.08


class ImgListDataset(Dataset):
    def __init__(self, root, imglist_path, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        if not os.path.exists(imglist_path):
            raise FileNotFoundError(f"Image list not found at {imglist_path}")

        with open(imglist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.root, path)
        try:
            img = Image.open(full_path).convert('RGB')
        except:
            return None, None, None

        if self.transform:
            img = self.transform(img)
        return img, label, path


class EnsembleCompetencyJudge(nn.Module):
    """
    Evaluates image competency using an Ensemble of ResNet50 and ViT-B/16.
    Includes internal normalization so it can accept [0, 1] inputs.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

        # Internal Normalization for ImageNet Models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print("Loading Ensemble (ResNet50 + ViT-B/16)...")
        self.resnet = models.resnet50(weights='IMAGENET1K_V1').eval().to(device)
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1').eval().to(device)

        self.energy_bias = -5.0

    def get_competency(self, img):
        """
        Args:
            img: Tensor [B, 3, H, W] in range [0, 1]
        """
        # 1. Normalize ON THE FLY (Corrects the mismatch)
        img_norm = self.normalize(img)

        with torch.no_grad():
            logits_r = self.resnet(img_norm)
            logits_v = self.vit(img_norm)

            # Ensemble Logits
            logits_ens = (logits_r + logits_v) / 2.0
            pred = logits_ens.argmax(dim=1).item()

            # MSP
            probs = F.softmax(logits_ens, dim=1)
            msp = probs.max().item()

            # Energy Score -> P(ID)
            energy = torch.logsumexp(logits_ens, dim=1).item()
            p_id = 1.0 / (1.0 + np.exp(-(energy + self.energy_bias)))

            parce = msp * p_id

            stats = {
                'msp': msp,
                'p_id': p_id,
                'energy': energy
            }
            return parce, pred, stats


def save_tensor_as_img(tensor, path):
    # Input is already [0, 1], so NO denormalization needed.
    img = torch.clamp(tensor, 0, 1)
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def sweep_and_select(judge, ae_model, img, label, output_dir, base_name):
    # 1. Generate Competency Mask (AE handles [0,1] internally)
    mask = generate_competency_mask(ae_model, img, percentile=0.40)

    # 2. Define Nuisances
    n_blur = LocalBlurNuisance(severity=1)
    n_blur.name = "blur"

    n_noise = LocalNoiseNuisance(severity=1)
    n_noise.name = "noise"

    n_pixel = LocalPixelationNuisance(severity=1)
    n_pixel.name = "pixel"

    n_spatial = LocalSpatialNuisance(severity=1)
    n_spatial.name = "spatial"

    n_bright = LocalPhotometricNuisance(mode='brightness', severity=1)
    n_bright.name = "brightness"

    n_contrast = LocalPhotometricNuisance(mode='contrast', severity=1)
    n_contrast.name = "contrast"

    n_sat = LocalPhotometricNuisance(mode='saturation', severity=1)
    n_sat.name = "saturation"

    nuisances = [n_blur, n_noise, n_pixel, n_spatial, n_bright, n_contrast, n_sat]

    selected_samples = []

    for nuisance in nuisances:
        buckets_needed = set(TARGET_LEVELS.keys())

        # Sweep param
        for p in np.linspace(0.05, 1.0, 40):
            if not buckets_needed: break

            # Apply Nuisance (Now operating on clean [0,1] data)
            img_perturbed = nuisance.apply(img, mask, manual_param=p)

            # Evaluate (Judge handles normalization)
            parce, pred, stats = judge.get_competency(img_perturbed)

            if pred == label:
                for level, target in TARGET_LEVELS.items():
                    if level in buckets_needed:
                        if abs(parce - target) < TOLERANCE:
                            # --- FOLDER STRUCTURE ---
                            nuisance_dir = os.path.join(output_dir, nuisance.name)
                            level_dir = os.path.join(nuisance_dir, str(level))
                            os.makedirs(level_dir, exist_ok=True)

                            fname = f"{base_name}.png"
                            save_path = os.path.join(level_dir, fname)

                            save_tensor_as_img(img_perturbed, save_path)

                            rel_path = os.path.join(nuisance.name, str(level), fname).replace("\\", "/")

                            selected_samples.append({
                                'path': rel_path,
                                'level': level,
                                'parce': parce,
                                'p': p,
                                'nuisance': nuisance.name,
                                'stats': stats
                            })

                            buckets_needed.remove(level)
                            break

    return selected_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="ImageNet root")
    parser.add_argument('--imglist', type=str, required=True, help="Path to .txt list")
    parser.add_argument('--ae_weights', type=str, required=True, help="Path to AE weights")
    parser.add_argument('--out_dir', type=str, default="./ln_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    judge = EnsembleCompetencyJudge(device)

    # AE
    ae = ClassifierAwareAE().to(device)
    ae.load_state_dict(torch.load(args.ae_weights, map_location=device))
    ae.eval()

    # Data - REMOVED NORMALIZATION
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Returns [0, 1]
    ])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_lines = []

    print(f"Starting Generation on {len(dataset)} images...")

    for i in tqdm(range(len(dataset))):
        img, label, path = dataset[i]
        if img is None: continue

        img = img.unsqueeze(0).to(device)
        base_name = os.path.splitext(os.path.basename(path))[0]

        parce, pred, _ = judge.get_competency(img)

        if pred == label and parce > 0.90:
            results = sweep_and_select(judge, ae, img, label, args.out_dir, base_name)

            for res in results:
                line = f"{res['path']} {label}"
                manifest_lines.append(line)

    with open(os.path.join(args.out_dir, "imagenet_ln_v2.txt"), "w") as f:
        f.write("\n".join(manifest_lines))

    print(f"Generation Complete. Output at {args.out_dir}")


if __name__ == "__main__":
    main()