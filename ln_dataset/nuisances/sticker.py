# ln_dataset/nuisances/sticker.py
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


def _parse_stickers_yaml_filenames(yaml_path: str) -> List[str]:
    """
    Robust parser:
    - Try PyYAML if available
    - Fallback: line-scan for `filename: ...`
    """
    filenames: List[str] = []
    try:
        import yaml  # type: ignore
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "stickers" in data and isinstance(data["stickers"], list):
            for it in data["stickers"]:
                if isinstance(it, dict) and "filename" in it:
                    filenames.append(str(it["filename"]))
        return filenames
    except Exception:
        pass

    # Fallback: simple scan
    try:
        with open(yaml_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("filename:"):
                    filenames.append(line.split(":", 1)[1].strip())
    except Exception:
        return []
    return filenames


@lru_cache(maxsize=1)
def _load_sticker_bank(stickers_dir: str, yaml_path: str) -> List[torch.Tensor]:
    """
    Returns list of sticker tensors on CPU, each [4,H,W] RGBA in [0,1].
    Cached so we don't reload per call.
    """
    files = _parse_stickers_yaml_filenames(yaml_path)
    bank: List[torch.Tensor] = []
    for fn in files:
        p = os.path.join(stickers_dir, fn)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGBA")
        t = TF.to_tensor(img)  # [4,H,W] in [0,1]
        bank.append(t)
    return bank


def _binary_erode(mask01: torch.Tensor, k: int) -> torch.Tensor:
    """
    mask01: [1,1,H,W] in {0,1}
    erosion via maxpool on inverse.
    """
    if k <= 1:
        return mask01
    if k % 2 == 0:
        k += 1
    inv = 1.0 - mask01
    dil_inv = F.max_pool2d(inv, kernel_size=k, stride=1, padding=k // 2)
    eroded = (dil_inv <= 1e-6).float()
    return eroded


class LocalStickerNuisance:
    """
    Photometric "sticker" overlay with mask-aware placement.
    - Chooses a random PNG with alpha from assets/stickers
    - Scales/rotates it
    - Places it strictly inside an eroded competency mask
    - Alpha-blends onto the image
    """

    def __init__(
        self,
        stickers_dir: str = "ln_dataset/assets/stickers",
        stickers_yaml: str = "ln_dataset/assets/stickers/stickers.yaml",
        min_size_frac: float = 0.07,
        max_size_frac: float = 0.28,
        min_alpha: float = 0.35,
        max_alpha: float = 0.90,
        mask_thresh: float = 0.35,
    ):
        self.stickers_dir = stickers_dir
        self.stickers_yaml = stickers_yaml
        self.min_size_frac = float(min_size_frac)
        self.max_size_frac = float(max_size_frac)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.mask_thresh = float(mask_thresh)

    def apply(self, img_tensor: torch.Tensor, mask: torch.Tensor, manual_param: Optional[float] = None, seed: Optional[int] = None):
        """
        img_tensor: [1,3,H,W] in [0,1]
        mask:       [1,1,H,W] in [0,1]
        manual_param p in [0,1] controls size/opacity
        """
        p = float(manual_param) if manual_param is not None else 0.0
        p = max(0.0, min(1.0, p))

        assert img_tensor.ndim == 4 and img_tensor.shape[0] == 1 and img_tensor.shape[1] == 3
        assert mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1
        _, _, H, W = img_tensor.shape
        device = img_tensor.device

        rng = np.random.default_rng(seed)

        bank = _load_sticker_bank(self.stickers_dir, self.stickers_yaml)
        if len(bank) == 0:
            # If assets missing, no-op (don’t silently fabricate—keeps benchmark honest).
            return img_tensor

        st = bank[int(rng.integers(0, len(bank)))]  # CPU [4,h,w]
        st_pil = TF.to_pil_image(st)  # RGBA

        # scale size with p
        base = min(H, W)
        size_frac = self.min_size_frac + (self.max_size_frac - self.min_size_frac) * p
        target = max(8, int(round(base * size_frac)))

        # keep aspect
        sw, sh = st_pil.size
        if sw >= sh:
            new_w = target
            new_h = max(8, int(round(target * (sh / max(1, sw)))))
        else:
            new_h = target
            new_w = max(8, int(round(target * (sw / max(1, sh)))))

        st_pil = st_pil.resize((new_w, new_h), resample=Image.BICUBIC)

        # small random rotation
        angle = float(rng.uniform(-25.0, 25.0))
        st_pil = st_pil.rotate(angle, resample=Image.BICUBIC, expand=True)

        st_rgba = TF.to_tensor(st_pil)  # [4,sh,sw]
        st_rgb = st_rgba[:3].to(device=device, dtype=img_tensor.dtype)
        st_a = st_rgba[3:4].to(device=device, dtype=img_tensor.dtype)  # [1,sh,sw]

        sh, sw = st_rgb.shape[1], st_rgb.shape[2]
        if sh >= H or sw >= W:
            return img_tensor  # too big, skip

        # Build feasible placement region: inside (eroded) binary mask
        mask_bin = (mask > self.mask_thresh).float()
        k = int(max(sh, sw))
        feasible = _binary_erode(mask_bin, k=k).squeeze(0).squeeze(0)  # [H,W]

        coords = torch.nonzero(feasible, as_tuple=False)
        if coords.numel() == 0:
            coords = torch.nonzero(mask_bin.squeeze(0).squeeze(0), as_tuple=False)
        if coords.numel() == 0:
            return img_tensor

        idx = int(rng.integers(0, coords.shape[0]))
        cy, cx = int(coords[idx, 0].item()), int(coords[idx, 1].item())

        top = int(np.clip(cy - sh // 2, 0, H - sh))
        left = int(np.clip(cx - sw // 2, 0, W - sw))

        out = img_tensor.clone()
        patch = out[:, :, top:top + sh, left:left + sw]
        mask_crop = mask[:, :, top:top + sh, left:left + sw].clamp(0, 1)

        # opacity scales with p; also multiply by mask_crop so mask is *actually* involved in blending too
        alpha = (self.min_alpha + (self.max_alpha - self.min_alpha) * p) * st_a[None, :, :, :] * mask_crop
        patch = patch * (1.0 - alpha) + st_rgb[None, :, :, :] * alpha
        out[:, :, top:top + sh, left:left + sw] = patch
        return out
