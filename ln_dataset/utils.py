import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import matplotlib.cm as cm

# --- CONSTANTS ---
STATS_FILE = "stats.json"
# Standard ImageNet Normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# ==========================================
# 1. DATASET HANDLING
# ==========================================
class ImgListDataset(Dataset):
    """
    Robust Dataset loader that handles:
    1. Lines with spaces in filenames (e.g. "Acura TL 2008/001.jpg 5")
    2. Corrupt images (skips them or returns None)
    """

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

                # Robust parsing for paths with spaces
                if len(parts) >= 2:
                    # The label is strictly the last item
                    label_str = parts[-1]
                    # The path is everything before it joined back together
                    path_str = " ".join(parts[:-1])

                    try:
                        label = int(label_str)
                        self.samples.append((path_str, label))
                    except ValueError:
                        print(f"Warning: Could not parse line: {line}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        full_path = os.path.join(self.root, rel_path)

        try:
            with open(full_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return None, label, rel_path

        if self.transform:
            img = self.transform(img)

        return img, label, rel_path


# ==========================================
# 2. LOSS FUNCTIONS
# ==========================================
def tv_weights(img, weight):
    """
    Total Variation Loss.
    Used to encourage spatial smoothness in generated masks/noise.
    """
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = weight * (h_variance + w_variance)
    return loss


# ==========================================
# 3. VISUALIZATION & DEBUGGING
# ==========================================
def _norm01(t):
    """Normalizes a tensor to [0, 1] for visualization."""
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def _to_pil_heat(map_tensor, cmap="jet"):
    """Converts a 1-channel tensor to a Heatmap PIL Image."""
    # Ensure it's on CPU and numpy
    if isinstance(map_tensor, torch.Tensor):
        m = _norm01(map_tensor).squeeze().detach().cpu().numpy()
    else:
        m = map_tensor

    # Apply colormap
    c = cm.get_cmap(cmap)
    rgb = (c(m)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def save_tensor_as_img(tensor, path):
    """Saves a (C,H,W) tensor as an image file."""
    # Undo normalization if it looks like it's in the [-2, 2] range (standard ImageNet)
    if tensor.min() < 0:
        mean = torch.tensor(NORM_MEAN).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(NORM_STD).view(3, 1, 1).to(tensor.device)
        tensor = tensor * std + mean

    save_image(tensor, path)


def save_debug_maps(competency_map, mask, original, path_prefix):
    """
    Saves a comprehensive set of debug images.
    1. Original Image
    2. Competency Map (Heatmap of where the model is confident)
    3. Selected Mask (Binary/Soft mask of the region being manipulated)
    4. Overlay (Competency Map superimposed on Original)
    """
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

    # 1. Save Raw Tensors
    save_tensor_as_img(original, f"{path_prefix}_original.png")
    save_tensor_as_img(mask, f"{path_prefix}_mask.png")

    # 2. Save Heatmaps
    comp_pil = _to_pil_heat(competency_map)
    comp_pil.save(f"{path_prefix}_competency_heatmap.png")

    # 3. Save Overlay
    # First, get original as PIL
    # (We temporarily save/load or do conversion manually)
    orig_path = f"{path_prefix}_original.png"
    if os.path.exists(orig_path):
        orig_pil = Image.open(orig_path).convert("RGB")

        # Resize heatmap to match original
        comp_pil = comp_pil.resize(orig_pil.size, resample=Image.BILINEAR)

        # Blend
        overlay = Image.blend(orig_pil, comp_pil, alpha=0.5)
        overlay.save(f"{path_prefix}_overlay.png")


# ==========================================
# 4. SALIENCY (Optional, for advanced debug)
# ==========================================
@torch.enable_grad()
def compute_input_saliency_msp(judge, img_tensor, target_label=None):
    """
    Computes gradient of the Maximum Softmax Probability w.r.t input.
    Useful to see which pixels contributed most to the 'competency'.
    """
    x = img_tensor.detach().clone().requires_grad_(True)

    # Forward pass through the Judge's primary model (usually ResNet)
    # Note: Judge normalizes internally usually, but we assume x is pre-normalized or handled by judge
    # Here we assume manual access to judge's backbone for gradients

    # Ideally, use the judge's built-in prediction method if accessible
    # This is a simplified placeholder.
    pass