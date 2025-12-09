import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from tqdm import tqdm
from PIL import Image

# Import Core Logic
from ln_dataset.core.msp_grad import get_msp_gradient
from ln_dataset.core.masks import generate_competency_mask
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance

# --- Configuration ---
# Scales: How close to the "breaking point" do we go?
# 0.2 = 20% of the way to breaking (Safe, high conf)
# 0.99 = 99% of the way to breaking (Safe, lowest conf)
SEVERITY_SCALES = [0.2, 0.4, 0.6, 0.8, 0.99]


class ImgListDataset(Dataset):
    """
    Reads data from an OpenOOD-style image list (path label).
    Returns (img, label, path) to preserve filenames.
    """

    def __init__(self, root, imglist_path, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        print(f"Loading image list from {imglist_path}...")
        with open(imglist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    path = parts[0]
                    label = int(parts[1])
                    self.samples.append((path, label))
        print(f"Found {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.root, path)

        # Open as RGB
        try:
            img = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            img = Image.new('RGB', (224, 224))  # Dummy fallback

        if self.transform:
            img = self.transform(img)

        return img, label, path


def get_nuisance_bounds(nuisance_obj):
    """
    Returns (safe_val, destroy_val, is_inverse)
    is_inverse=True means 1.0 is Clean, 0.0 is Destroyed.
    """
    if isinstance(nuisance_obj, LocalPixelationNuisance):
        return 1.0, 0.05, True
    elif isinstance(nuisance_obj, LocalPhotometricNuisance):
        if nuisance_obj.mode in ['contrast', 'saturation']:
            return 1.0, 0.0, True
        else:  # Brightness
            return 0.0, 1.0, False
    elif isinstance(nuisance_obj, LocalSpatialNuisance):
        return 0.0, 200.0, False
    elif isinstance(nuisance_obj, LocalNoiseNuisance):
        return 0.0, 0.5, False  # 0.5 noise is usually enough to destroy anything
    else:
        return 0.0, 1.0, False


def find_breaking_point(model, img, label, nuisance_obj, normalize, mask):
    """
    Binary search for the parameter that flips the prediction.
    """
    safe_p, fail_p, is_inverse = get_nuisance_bounds(nuisance_obj)

    # 1. Quick Check: Does the 'fail' condition actually flip the label?
    if isinstance(nuisance_obj, LocalNoiseNuisance):
        _, grads = get_msp_gradient(model, normalize(img), label)
        out_fail = nuisance_obj.apply(img, mask, gradient_tensor=grads, manual_param=fail_p)
    else:
        out_fail = nuisance_obj.apply(img, mask, manual_param=fail_p)

    with torch.no_grad():
        if model(normalize(out_fail)).argmax(1).item() == label:
            # The image is a "Rock" - even max destruction doesn't break it.
            return fail_p

    # 2. Binary Search
    current_safe = safe_p
    current_fail = fail_p

    for _ in range(8):  # Precision iterations
        mid = (current_safe + current_fail) / 2

        if isinstance(nuisance_obj, LocalNoiseNuisance):
            _, grads = get_msp_gradient(model, normalize(img), label)
            out = nuisance_obj.apply(img, mask, gradient_tensor=grads, manual_param=mid)
        else:
            out = nuisance_obj.apply(img, mask, manual_param=mid)

        with torch.no_grad():
            pred = model(normalize(out)).argmax(1).item()

        if pred == label:
            current_safe = mid
        else:
            current_fail = mid

    return current_safe


def generate_dataset(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = models.resnet50(weights='IMAGENET1K_V1').eval().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImgListDataset(args.data, args.imglist, transform=transform)
    # Note: batch_size=1 is required for this specific logic
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    nuisances = {
        'brightness': LocalPhotometricNuisance('brightness', 1),
        'contrast': LocalPhotometricNuisance('contrast', 1),
        'saturation': LocalPhotometricNuisance('saturation', 1),
        'noise': LocalNoiseNuisance(1),
        'pixel': LocalPixelationNuisance(1),
        'spatial': LocalSpatialNuisance(1)
    }

    img_list_buffers = {}  # Key: "nuisance_severity", Value: list of strings

    print("Starting Adaptive Generation...")

    for i, (img, label, rel_path) in enumerate(tqdm(loader)):
        img = img.to(device)
        label_item = label.item()
        rel_path = rel_path[0]  # Unpack from batch tuple

        # 1. Check Baseline
        with torch.no_grad():
            base_pred = model(normalize(img)).argmax(1).item()

        is_correct = (base_pred == label_item)

        if is_correct:
            saliency, grads = get_msp_gradient(model, normalize(img), label_item)
            mask = generate_competency_mask(saliency, severity=3)
        else:
            # Mask that does nothing (or minimal) if already incorrect
            mask = torch.zeros_like(img[:, 0:1, :, :])

        for n_name, n_obj in nuisances.items():

            # Find Breaking Point
            safe_val, destroy_val, is_inverse = get_nuisance_bounds(n_obj)

            if is_correct:
                breaking_p = find_breaking_point(model, img, label_item, n_obj, normalize, mask)
            else:
                breaking_p = safe_val

                # Generate 5 Levels
            for level_idx, scale in enumerate(SEVERITY_SCALES):
                severity_level = level_idx + 1

                # Interpolate
                delta = breaking_p - safe_val
                p = safe_val + (delta * scale)

                # Apply
                if isinstance(n_obj, LocalNoiseNuisance) and is_correct:
                    _, grads_new = get_msp_gradient(model, normalize(img), label_item)
                    out = n_obj.apply(img, mask, gradient_tensor=grads_new, manual_param=p)
                else:
                    out = n_obj.apply(img, mask, manual_param=p)

                # --- FILE SAVING LOGIC (FLATTENED) ---

                # Create directory: output/nuisance/severity/
                # We do NOT include os.path.dirname(rel_path) here.
                save_dir = os.path.join(args.output_root, n_name, str(severity_level))
                os.makedirs(save_dir, exist_ok=True)

                # Filename: Just the basename (e.g., ILSVRC2012_val_00000293.png)
                original_fname = os.path.basename(rel_path)
                final_filename = os.path.splitext(original_fname)[0] + ".png"
                save_path = os.path.join(save_dir, final_filename)

                transforms.ToPILImage()(out.squeeze().cpu()).save(save_path)

                # Add to List Buffer
                # OpenOOD format: relative/path/from/dataset/root label
                # Result: nuisance/severity/image.png

                list_key = f"{n_name}_{severity_level}"
                if list_key not in img_list_buffers:
                    img_list_buffers[list_key] = []

                entry_rel_path = os.path.join(n_name, str(severity_level), final_filename)
                entry_rel_path = entry_rel_path.replace("\\", "/")  # Ensure forward slashes

                img_list_buffers[list_key].append(f"{entry_rel_path} {label_item}")

    # Write Image Lists
    os.makedirs(args.list_root, exist_ok=True)
    print("Writing Image Lists...")
    for key, lines in img_list_buffers.items():
        fname = os.path.join(args.list_root, f"{key}.txt")
        with open(fname, 'w') as f:
            f.write("\n".join(lines))

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Root of source images")
    parser.add_argument('--imglist', type=str, required=True, help="Path to source image list")
    parser.add_argument('--output_root', type=str, default="data/images_largescale/imagenet_ln",
                        help="Where to save generated images")
    parser.add_argument('--list_root', type=str, default="data/benchmark_imglist/imagenet/test_imagenet_ln",
                        help="Where to save new image lists")

    args = parser.parse_args()
    generate_dataset(args)