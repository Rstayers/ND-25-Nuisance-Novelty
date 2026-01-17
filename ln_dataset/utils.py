import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import models
import matplotlib.pyplot as plt

# --- CONSTANTS ---
STATS_FILE = "stats.json"
WEIGHTS_VARIANT = "IMAGENET1K_V1"


# ==========================================
# 1. DATASET HANDLING
# ==========================================
class ImgListDataset(Dataset):
    def __init__(self, root, imglist_path, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        if not os.path.exists(imglist_path):
            raise FileNotFoundError(f"List not found: {imglist_path}")

        with open(imglist_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2: continue
                path = " ".join(parts[:-1])
                label = int(parts[-1])
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.root, path)
        try:
            img = Image.open(full_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label, path
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, label, path


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def tv_weights(name: str):
    name = name.lower()
    if name == "resnet50":
        return getattr(models.ResNet50_Weights, WEIGHTS_VARIANT)
    if name in ["vit_b_16", "vitb16"]:
        return getattr(models.ViT_B_16_Weights, WEIGHTS_VARIANT)
    if name in ["convnext_t", "convnext_tiny"]:
        return getattr(models.ConvNeXt_Tiny_Weights, WEIGHTS_VARIANT)
    if name in ["densenet121", "densenet"]:
        return getattr(models.DenseNet121_Weights, WEIGHTS_VARIANT)
    raise ValueError(name)


def save_tensor_as_img(tensor, path):
    """
    Saves a (C, H, W) tensor to an image file.
    Assumes tensor is [0, 1] (no extra denormalization applied).
    """
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    # Clamp to ensure valid range before saving
    tensor = torch.clamp(tensor, 0, 1)
    img = TF.to_pil_image(tensor)
    img.save(path)


def _save_heatmap(tensor, path, cmap='jet'):
    """Helper to save a single channel tensor as a heatmap."""
    tensor = tensor.detach().cpu()
    while tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    # vmin/vmax ensure 0 is Blue/Black and 1 is Red/White
    plt.imsave(path, tensor.numpy(), cmap=cmap, vmin=0, vmax=1)


# ==========================================
# 3. DEBUG MAPS VISUALIZATION
# ==========================================
def save_debug_maps(competency_map, mask, img_tensor, path_prefix):
    """
    Arguments aligned with generate_ln.py call:
    1. competency_map (Heatmap)
    2. mask (Binary)
    3. img_tensor (Original Image)
    4. path_prefix (Output path base)
    """
    # 1. Save Original
    # We strip the extension from path_prefix if it was inadvertently added,
    # though usually path_prefix is "dir/filename_base"
    save_tensor_as_img(img_tensor, f"{path_prefix}_original.png")

    # 2. Save Competency Heatmap (Magma style from reference)
    # Used for familiarity/competency visualization
    _save_heatmap(competency_map, f"{path_prefix}_competency_heatmap.png", cmap='magma')

    # 3. Save Binary Mask
    _save_heatmap(mask, f"{path_prefix}_mask.png", cmap='gray')

    # 4. Save Overlay (Green Tint Logic)
    # Prepare Image (H, W, 3) numpy
    img_np = img_tensor.detach().cpu()
    if img_np.ndim == 4: img_np = img_np.squeeze(0)
    img_np = img_np.permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
    img_np = np.clip(img_np, 0, 1)  # Ensure 0-1

    # Prepare Mask (H, W) numpy
    mask_np = mask.detach().cpu().squeeze().numpy()

    # Create Overlay
    overlay = img_np.copy()
    # Apply green tint to masked regions [Channel 1 is Green]
    # Logic: pixels where mask > 0.1 get +0.3 brightness in Green channel
    overlay[mask_np > 0.1, 1] = np.clip(overlay[mask_np > 0.1, 1] + 0.3, 0, 1)

    plt.imsave(f"{path_prefix}_overlay.png", overlay)


