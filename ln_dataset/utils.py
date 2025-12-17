# ==========================================
# DATA & UTILS
# ==========================================
import json
import os

import numpy as np
import torch
from PIL import Image
from regex import F
from torch.utils.data import Dataset

from ln_dataset.core.autoencoder import get_reconstruction_error
STATS_FILE = "ln_dataset/assets/parce_stats_full.pt"
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Global Constants
STATS_FILE = "ln_dataset/assets/parce_stats_full.pt"
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


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



def save_tensor_as_img(tensor, path):
    """Saves a (C, H, W) tensor to an image file."""
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4: tensor = tensor.squeeze(0)
    img = TF.to_pil_image(tensor)
    img.save(path)


def _save_heatmap(tensor, path, cmap='jet'):
    """Helper to save a single channel tensor as a heatmap."""
    tensor = tensor.detach().cpu().squeeze()
    plt.imsave(path, tensor.numpy(), cmap=cmap, vmin=0, vmax=1)


def save_debug_maps(judge, ae, img, label, mask, output_dir):
    from ln_dataset.core.autoencoder import get_reconstruction_error
    os.makedirs(output_dir, exist_ok=True)
    save_tensor_as_img(img, os.path.join(output_dir, "00_orig.png"))
    err = get_reconstruction_error(ae, img)
    err = (err - err.min()) / (err.max() - err.min() + 1e-6)
    _save_heatmap(err, os.path.join(output_dir, "01_familiarity.png"), cmap='magma')
    _save_heatmap(mask, os.path.join(output_dir, "03_final_mask.png"), cmap='gray')
    img_np = img.permute(0, 2, 3, 1).cpu().squeeze().numpy()
    mask_np = mask.cpu().squeeze().numpy()
    overlay = img_np.copy()
    overlay[mask_np > 0.1, 1] = np.clip(overlay[mask_np > 0.1, 1] + 0.3, 0, 1)
    plt.imsave(os.path.join(output_dir, "04_overlay.png"), overlay)


# ==========================================
#  MASK DEBUG
# ==========================================

def _norm01(x, eps=1e-8):
    x = x.float()
    x = x - x.min()
    x = x / (x.max() + eps)
    return x

def _to_pil_rgb(img_tensor_1x3xhxw):
    img = torch.clamp(img_tensor_1x3xhxw, 0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def _to_pil_gray(map_1x1xhxw):
    m = _norm01(map_1x1xhxw).squeeze().detach().cpu().numpy()
    m = (m * 255).astype(np.uint8)
    return Image.fromarray(m, mode="L")

def _to_pil_heat(map_1x1xhxw, cmap="magma"):
    m = _norm01(map_1x1xhxw).squeeze().detach().cpu().numpy()
    try:
        import matplotlib.cm as cm
        c = cm.get_cmap(cmap)
        rgb = (c(m)[..., :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb)
    except Exception:
        # fallback: grayscale
        return _to_pil_gray(map_1x1xhxw).convert("RGB")

def _overlay(pil_rgb, map_1x1xhxw, alpha=0.5, cmap="magma"):
    heat = _to_pil_heat(map_1x1xhxw, cmap=cmap).resize(pil_rgb.size)
    return Image.blend(pil_rgb, heat, alpha=alpha)

@torch.enable_grad()
def compute_input_saliency_msp(judge, img_tensor, target_label=None):
    """
    Gradient magnitude of ensemble MSP for target_label (or predicted class if None).
    Returns: (1,1,H,W)
    """
    x = img_tensor.detach().clone().requires_grad_(True)

    # same normalization path judge uses :contentReference[oaicite:3]{index=3}
    x_norm = judge.normalize(x)
    logits_r = judge.resnet(x_norm)
    logits_v = judge.vit(x_norm)
    probs = (F.softmax(logits_r, dim=1) + F.softmax(logits_v, dim=1)) / 2.0

    cls = int(target_label) if target_label is not None else int(probs.argmax(dim=1).item())
    score = probs[0, cls]

    judge.resnet.zero_grad(set_to_none=True)
    judge.vit.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()

    score.backward()
    sal = x.grad.detach().abs().mean(dim=1, keepdim=True)
    return sal


