import torch
import torch.nn as nn
import torch.nn.functional as F
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
from ln_dataset.core.masks import generate_ensemble_saliency_mask

# Nuisance Modules
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance

# --- CONFIGURATION (OLD BIN LOGIC) ---
TARGET_LEVELS = {
    '1': 0.95,  # Near perfect
    '2': 0.80,  # Good but uncertain
    '3': 0.60,  # Ambiguous
    '4': 0.40,  # Poor
    '5': 0.25  # Very Low Competency
}
TOLERANCE = 0.12
DEBUG_MAX_IMAGES = 20


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


class EnsembleParceJudge(nn.Module):
    """
    Evaluates competency using the PaRCE method (MSP * P_ID) on an ENSEMBLE.
    Keeps accuracy high (Robustness check) while degrading PaRCE.
    """

    def __init__(self, device, ae_model):
        super().__init__()
        self.device = device
        self.ae_model = ae_model

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print("Loading Ensemble (ResNet50 + ViT-B/16)...")
        self.resnet = models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()

        # Calibration for P_ID (Adjusted to be stricter)
        self.error_mu = 0.02
        self.error_sigma = 0.02

    @torch.no_grad()
    def evaluate_step(self, img, label):
        """
        Returns:
            is_robust (bool): True ONLY if BOTH models are correct.
            parce_score (float): Ensemble PaRCE score.
        """
        img_norm = self.normalize(img)

        # 1. Forward Pass Both Models
        logits_r = self.resnet(img_norm)
        logits_v = self.vit(img_norm)

        # 2. Check Robustness (Accuracy)
        pred_r = logits_r.argmax(dim=1).item()
        pred_v = logits_v.argmax(dim=1).item()
        is_robust = (pred_r == label) and (pred_v == label)

        # 3. Calculate Ensemble MSP
        logits_ens = (logits_r + logits_v) / 2.0
        probs_ens = F.softmax(logits_ens, dim=1)
        msp_ens = probs_ens.max().item()

        # 4. Calculate P_ID (Familiarity via AE Reconstruction)
        error_map = get_reconstruction_error(self.ae_model, img)
        mean_error = error_map.mean().item()
        p_id = np.exp(- (mean_error) / self.error_mu)
        p_id = np.clip(p_id, 0.0, 1.0)

        # 5. Final PaRCE Score
        parce_score = msp_ens * p_id

        return is_robust, parce_score


def save_tensor_as_img(tensor, path):
    img = torch.clamp(tensor, 0, 1)
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def sweep_and_select(judge, ae_model, img, label, output_dir, base_name, do_debug=False):
    # 1. Generate New PaRCE Mask (Reconstruction Only)
    # Using percentile=0.60 to cover more familiar areas for better degradation
    mask = generate_ensemble_saliency_mask(judge.resnet, judge.vit, img, label)

    # Visualize Debug
    if do_debug:
        debug_dir = os.path.join(output_dir, "debug_viz")
        os.makedirs(debug_dir, exist_ok=True)

        # Red Overlay for Mask
        overlay_alpha = 0.6
        solid_red = torch.zeros_like(img);
        solid_red[:, 0, :, :] = 1.0
        mask_viz = img * (1 - overlay_alpha * mask) + solid_red * (overlay_alpha * mask)

        save_image(mask_viz, os.path.join(debug_dir, f"{base_name}_saliency_mask.jpg"))

    nuisances = [
        LocalNoiseNuisance(severity=1),
        LocalPixelationNuisance(severity=1),
        LocalSpatialNuisance(severity=1),
        LocalPhotometricNuisance(mode='brightness', severity=1),
        LocalPhotometricNuisance(mode='contrast', severity=1),
        LocalPhotometricNuisance(mode='saturation', severity=1)
    ]
    names = ["noise", "pixel", "spatial", "brightness", "contrast", "saturation"]
    for n, name in zip(nuisances, names): n.name = name

    selected_samples = []

    for nuisance in nuisances:
        buckets_needed = set(TARGET_LEVELS.keys())

        # Sweep parameter p from 0.05 to 1.0
        for p in np.linspace(0.05, 1.0, 40):
            if not buckets_needed: break

            img_perturbed = nuisance.apply(img, mask, manual_param=p)

            # Evaluate with updated PaRCE Judge
            is_robust, parce = judge.evaluate_step(img_perturbed, label)

            # --- OLD BIN LOGIC with NEW ROBUSTNESS CHECK ---
            # We require the image to be Robust (Top-1 Correct) to be included.
            if is_robust:
                for level, target in TARGET_LEVELS.items():
                    if level in buckets_needed:
                        if abs(parce - target) < TOLERANCE:
                            # Save Logic
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
                                'nuisance': nuisance.name
                            })

                            buckets_needed.remove(level)
                            break
            else:
                # If robustness fails, we stop this nuisance sweep immediately
                # (since increasing p will likely only make it worse)
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

    ae = ClassifierAwareAE().to(device)
    ae.load_state_dict(torch.load(args.ae_weights, map_location=device))
    ae.eval()

    # New PaRCE Judge
    judge = EnsembleParceJudge(device, ae_model=ae)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_lines = []
    debug_counter = 0

    print(f"Starting Generation on {len(dataset)} images...")

    for i in tqdm(range(len(dataset))):
        img, label, path = dataset[i]
        if img is None: continue

        img = img.unsqueeze(0).to(device)
        base_name = os.path.splitext(os.path.basename(path))[0]

        is_robust, parce = judge.evaluate_step(img, label)

        # Start with High Competency images
        if parce > 0.85 and is_robust:
            do_debug = (debug_counter < DEBUG_MAX_IMAGES)
            results = sweep_and_select(judge, ae, img, label, args.out_dir, base_name, do_debug=do_debug)
            if do_debug: debug_counter += 1

            for res in results:
                line = f"{res['path']} {label}"
                manifest_lines.append(line)

    with open(os.path.join(args.out_dir, "imagenet_ln_v2.txt"), "w") as f:
        f.write("\n".join(manifest_lines))
    print(f"Generation Complete. Output at {args.out_dir}")


if __name__ == "__main__":
    main()