import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

# Core Logic
from ln_dataset.core.autoencoder import SimpleConvAE, get_reconstruction_error
from ln_dataset.core.masks import generate_competency_mask
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance
from ln_dataset.nuisances.blur import LocalBlurNuisance

# --- CONFIGURATION ---
TARGET_LEVELS = {
    '1': 0.95,
    '2': 0.80,
    '3': 0.60,
    '4': 0.40,
    '5': 0.15
}
MSP_TOLERANCE = 0.12


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
            print(f"Error loading {full_path}, skipping.")
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)
        return img, label, path


def get_msp(model, img_tensor):
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        msp, preds = probs.max(dim=1)
    return msp.item(), preds.item()


def get_nuisance_bounds(nuisance_obj):
    if isinstance(nuisance_obj, LocalPixelationNuisance):
        return 1.0, 0.02
    elif isinstance(nuisance_obj, LocalPhotometricNuisance):
        if nuisance_obj.mode == 'brightness':
            return 0.0, 0.6
        elif nuisance_obj.mode == 'contrast':
            return 1.0, 0.1
        elif nuisance_obj.mode == 'saturation':
            return 1.0, 4.0
    elif isinstance(nuisance_obj, LocalSpatialNuisance):
        return 0.0, 120.0
    elif isinstance(nuisance_obj, LocalNoiseNuisance):
        return 0.0, 1.5
    elif isinstance(nuisance_obj, LocalBlurNuisance):
        return 0.0, 5.0
    return 0.0, 1.0


def sweep_and_select(model, img, label, nuisance, mask, normalize):
    start_p, end_p = get_nuisance_bounds(nuisance)
    steps = 50
    candidates = []

    for i in range(steps):
        alpha = i / (steps - 1)
        p = start_p + (end_p - start_p) * alpha

        out = nuisance.apply(img, mask, manual_param=p)
        msp, pred = get_msp(model, normalize(out))

        if pred == label:
            candidates.append({'msp': msp, 'p': p, 'img': out})

    if not candidates:
        return {}

    selected_levels = {}
    for level_key, target_msp in TARGET_LEVELS.items():
        best_cand = None
        min_diff = float('inf')

        for cand in candidates:
            diff = abs(cand['msp'] - target_msp)
            if diff < min_diff:
                min_diff = diff
                best_cand = cand

        if best_cand and min_diff < MSP_TOLERANCE:
            selected_levels[level_key] = best_cand

    return selected_levels


def generate_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Models
    model = models.resnet50(weights='IMAGENET1K_V1').eval().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ae = SimpleConvAE().to(device).eval()
    # Load AE Weights
    weights_path = "./ln_dataset/assets/ae_competency_weights.pth"
    if os.path.exists(weights_path):
        ae.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded Competency Weights from {weights_path}")
    else:
        print("WARNING: AE weights not found! Using random initialization.")

    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Nuisances
    nuisances = {
        'brightness': LocalPhotometricNuisance('brightness', 1),
        'contrast': LocalPhotometricNuisance('contrast', 1),
        'saturation': LocalPhotometricNuisance('saturation', 1),
        'noise': LocalNoiseNuisance(1),
        'pixel': LocalPixelationNuisance(1),
        'spatial': LocalSpatialNuisance(1),
        'blur': LocalBlurNuisance(1)
    }

    # Buffers for output files
    clean_lines = []
    meta_lines = []

    print("Starting 5-Level Stratified Generation...")

    for i, (img, label, rel_path) in enumerate(tqdm(loader)):
        img = img.to(device)
        label_item = label.item()
        rel_path = rel_path[0]

        # Baseline Check
        msp_clean, pred_clean = get_msp(model, normalize(img))
        if pred_clean != label_item:
            continue

            # Generate Competency Mask
        mask = generate_competency_mask(ae, img, percentile=0.6)

        for n_name, n_obj in nuisances.items():
            results = sweep_and_select(model, img, label_item, n_obj, mask, normalize)

            for level_key, data in results.items():
                # Save Structure: output_root/nuisance_name/1/filename.png
                save_dir = os.path.join(args.output_root, n_name, level_key)
                os.makedirs(save_dir, exist_ok=True)

                orig_fname = os.path.basename(rel_path)
                # Ensure we save as PNG to prevent compression artifacts
                final_fname = os.path.splitext(orig_fname)[0] + ".png"
                save_path = os.path.join(save_dir, final_fname)

                # Save Image
                img_to_save = data['img'].detach().cpu().squeeze()
                img_to_save = torch.clamp(img_to_save, 0, 1)
                transforms.ToPILImage()(img_to_save).save(save_path)

                # --- MANIFEST GENERATION ---
                # Build the path relative to the "virtual" root using the provided prefix
                # Example prefix: images_largescale\imagenet_ln
                # Example structure: prefix \ nuisance \ level \ filename

                if args.manifest_prefix:
                    # Use os.path.join for safety, but check separator preference
                    virtual_path = os.path.join(args.manifest_prefix, n_name, level_key, final_fname)
                    # If user explicitly put backslashes in prefix, ensure consistency
                    if '\\' in args.manifest_prefix:
                        virtual_path = virtual_path.replace('/', '\\')
                else:
                    virtual_path = os.path.join(n_name, level_key, final_fname)

                # 1. Clean Line (Standard format: path label)
                clean_lines.append(f"{virtual_path} {label_item}")

                # 2. Metadata Line (path label msp param)
                meta_lines.append(f"{virtual_path} {label_item} {data['msp']:.4f} {data['p']:.4f}")

    # Write Manifests
    os.makedirs(args.list_root, exist_ok=True)

    # Main Manifest (Loadable by PyTorch)
    clean_path = os.path.join(args.list_root, "imagenet_ln.txt")
    with open(clean_path, 'w') as f:
        f.write("\n".join(clean_lines))

    # Metadata Manifest (For your Analysis Suite)
    meta_path = os.path.join(args.list_root, "imagenet_ln_meta.txt")
    with open(meta_path, 'w') as f:
        f.write("\n".join(meta_lines))

    print(f"Done.")
    print(f"Loader List:   {clean_path}")
    print(f"Analysis Data: {meta_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Root of source images")
    parser.add_argument('--imglist', type=str, required=True, help="Path to source image list")
    parser.add_argument('--output_root', type=str, default="data/images_largescale/imagenet_ln")
    parser.add_argument('--list_root', type=str, default="data/benchmark_imglist/imagenet_ln")
    parser.add_argument('--manifest_prefix', type=str, default="imagenet_ln", help="Prefix to prepend to paths in the text file")

    args = parser.parse_args()
    generate_dataset(args)