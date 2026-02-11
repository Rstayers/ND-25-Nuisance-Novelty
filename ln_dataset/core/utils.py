import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.segmentation import felzenszwalb, mark_boundaries  # <--- NEW DEPENDENCY
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


def _tensor_to_numpy(tensor):
    """Helper to convert any tensor (B,C,H,W or C,H,W) to (H,W,C) numpy."""
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    if t.ndim == 3 and t.shape[0] in [1, 3]:
        t = t.permute(1, 2, 0)  # C,H,W -> H,W,C
    elif t.ndim == 2:
        pass  # Already H,W
    return t.numpy()


import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib import cm

# Removed skimage dependency (felzenszwalb, mark_boundaries) to avoid "yellow outlines"
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


def _tensor_to_numpy(tensor):
    """Helper to convert any tensor (B,C,H,W or C,H,W) to (H,W,C) numpy."""
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    if t.ndim == 3 and t.shape[0] in [1, 3]:
        t = t.permute(1, 2, 0)  # C,H,W -> H,W,C
    elif t.ndim == 2:
        pass  # Already H,W
    return t.numpy()


def save_debug_maps(img, recon_error, mask, save_path):
    """
    Saves a 4-panel debug image:
    1. Original Image
    2. Reconstruction Error Heatmap
    3. Final Selected Mask (Solo)
    4. Mask Overlay (Original + Red Highlight)
    """

    # --- 1. Prepare Original Image ---
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu()
        if img_np.ndim == 4: img_np = img_np.squeeze(0)
        img_np = img_np.permute(1, 2, 0).numpy()
    else:
        img_np = img
    # Normalize to [0, 1]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # --- 2. Prepare Reconstruction Error (Heatmap) ---
    if isinstance(recon_error, torch.Tensor):
        err_np = recon_error.detach().cpu().squeeze().numpy()
    else:
        err_np = recon_error
    # Normalize error map for better contrast
    err_viz = (err_np - err_np.min()) / (err_np.max() - err_np.min() + 1e-8)

    # --- 3. Prepare Mask ---
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().squeeze().numpy()
    else:
        mask_np = mask

    # --- 4. Prepare Overlay ---
    # Create a red overlay: Where mask is 1, blend image with red
    overlay_np = img_np.copy()
    red_layer = np.zeros_like(img_np)
    red_layer[:, :, 0] = 1.0  # Red channel max

    # Define opacity for the mask
    alpha = 0.5
    # Create boolean mask
    mask_bool = mask_np > 0.5

    # Blend only where mask is active
    overlay_np[mask_bool] = (overlay_np[mask_bool] * (1 - alpha)) + (red_layer[mask_bool] * alpha)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Reconstruction Error
    # using 'magma' for high contrast
    im_err = axes[1].imshow(err_viz, cmap='magma')
    axes[1].set_title("Reconstruction Error", fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Mask (Solo)
    axes[2].imshow(mask_np, cmap='gray')
    axes[2].set_title("Selected Mask", fontsize=16, fontweight='bold')
    axes[2].axis('off')

    # Panel 4: Mask Overlay
    axes[3].imshow(overlay_np)
    axes[3].set_title("Mask Overlay", fontsize=16, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Force extension
    if not save_path.endswith('.png'):
        save_path += '.png'

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)