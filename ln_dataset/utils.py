# ==========================================
# DATA & UTILS
# ==========================================
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ln_dataset.core.autoencoder import get_reconstruction_error
STATS_FILE = "ln_dataset/assets/parce_stats_full.pt"
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

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

def save_debug_maps(judge, ae_model, img, label, mask, debug_dir):
    os.makedirs(debug_dir, exist_ok=True)

    pil_img = _to_pil_rgb(img)
    pil_img.save(os.path.join(debug_dir, "img.png"))

    # Reconstruction error map (your mask code is based on recon error in the simple version :contentReference[oaicite:4]{index=4})
    err = get_reconstruction_error(ae_model, img)  # (1,1,H,W)
    fam = 1.0 - _norm01(err)

    sal = compute_input_saliency_msp(judge, img, target_label=label)

    # Save raw/heat/overlay for each
    _to_pil_gray(err).save(os.path.join(debug_dir, "err_gray.png"))
    _to_pil_heat(err, "magma").save(os.path.join(debug_dir, "err_heat.png"))
    _overlay(pil_img, err, 0.5, "magma").save(os.path.join(debug_dir, "err_overlay.png"))

    _to_pil_gray(fam).save(os.path.join(debug_dir, "fam_gray.png"))
    _to_pil_heat(fam, "viridis").save(os.path.join(debug_dir, "fam_heat.png"))
    _overlay(pil_img, fam, 0.5, "viridis").save(os.path.join(debug_dir, "fam_overlay.png"))

    _to_pil_gray(sal).save(os.path.join(debug_dir, "sal_gray.png"))
    _to_pil_heat(sal, "inferno").save(os.path.join(debug_dir, "sal_heat.png"))
    _overlay(pil_img, sal, 0.5, "inferno").save(os.path.join(debug_dir, "sal_overlay.png"))

    _to_pil_gray(mask).save(os.path.join(debug_dir, "mask_gray.png"))
    _to_pil_heat(mask, "plasma").save(os.path.join(debug_dir, "mask_heat.png"))
    _overlay(pil_img, mask, 0.5, "plasma").save(os.path.join(debug_dir, "mask_overlay.png"))

    # Also save a binarized mask snapshot
    mask_bin = (mask > 0.5).float()
    _to_pil_gray(mask_bin).save(os.path.join(debug_dir, "mask_bin.png"))

    stats = {
        "label": int(label),
        "err_minmax": [float(err.min().item()), float(err.max().item())],
        "sal_minmax": [float(sal.min().item()), float(sal.max().item())],
        "mask_mean": float(mask.mean().item()),
        "mask_bin_area": float(mask_bin.mean().item()),
    }
    with open(os.path.join(debug_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
