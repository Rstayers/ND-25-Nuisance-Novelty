# imagenet_c_psycho/psycho_imglist_dataset.py
import os
import re
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

_SEV_RE = re.compile(r'(?:/|\\)(?:severity[_-]?([1-5])|([1-5]))(?:/|\\)')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _read_imglist(imglist_file: str) -> List[Tuple[str, int]]:
    out = []
    with open(imglist_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "<relpath> <label>"
            rel, lab = line.rsplit(' ', 1)
            out.append((rel, int(lab)))
    return out

def _severity_from_rel(relpath: str):
    """
    Matches both .../<corruption>/3/... and .../<corruption>/severity_3/...
    Returns int in [1..5] or None for clean ImageNet.
    """
    m = _SEV_RE.search(relpath)
    if not m:
        return None
    if m.group(1):
        return int(m.group(1))
    if m.group(2):
        return int(m.group(2))
    return None

def _sev_to_weight(sev: int, eps: float = 0.05) -> float:
    # Harder (higher severity) => penalize less:
    # sev∈{1..5} -> w∈[eps,1], linear: w = eps + (1-eps)*(5-sev)/4
    return eps + (1.0 - eps) * (5 - sev) / 4.0

class PsychoImglistDataset(Dataset):
    """
    Standalone dataset:
      - expects split_cfg with keys:
          data_dir, imglist_pth, batch_size, shuffle, and optional:
          inetc_weighting: {epsilon: float}
      - returns dict with keys: data (tensor), label (int), impath (str), weight (float32)
    """

    def __init__(self, split_cfg: Dict, split_name: str = "train"):
        self.split_name = split_name
        self.data_dir = split_cfg["data_dir"]
        self.imglist_pth = split_cfg["imglist_pth"]
        self.samples = _read_imglist(self.imglist_pth)

        iw = split_cfg.get("inetc_weighting", {}) or {}
        self.eps = float(iw.get("epsilon", 0.05))

        # defaults
        pre_size = split_cfg.get("pre_size", 256)
        image_size = split_cfg.get("image_size", 224)

        # optional augment section from top-level YAML (passed down by pipeline)
        # if not present, we still default to eval transforms on non-train.
        aug = split_cfg.get("__augment__", {})  # pipeline will inject this

        if self.split_name == "train":
            tfs = []
            if aug.get("random_resized_crop", True):
                from torchvision.transforms import InterpolationMode
                tfs.append(
                    transforms.RandomResizedCrop(image_size, interpolation=transforms.InterpolationMode.BILINEAR))
            else:
                tfs += [
                    transforms.Resize(pre_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(image_size),
                ]
            if aug.get("random_horizontal_flip", True):
                tfs.append(transforms.RandomHorizontalFlip())

            ra = aug.get("randaugment", None)
            if ra is not None:
                tfs.append(transforms.RandAugment(num_ops=int(ra.get("num_ops", 2)),
                                                  magnitude=int(ra.get("magnitude", 9))))

            tfs += [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform = transforms.Compose(tfs)
        else:
            # eval transforms
            self.transform = transforms.Compose([
                transforms.Resize(pre_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel, label = self.samples[idx]
        impath = os.path.join(self.data_dir, rel).replace('\\', '/')
        with Image.open(impath) as img:
            img = img.convert("RGB")
            x = self.transform(img)

        sev = _severity_from_rel(rel)
        if sev is None:
            w = 1.0  # clean ImageNet
        else:
            w = _sev_to_weight(sev, self.eps)

        return {
            "data": x,
            "label": torch.tensor(label, dtype=torch.long),
            "impath": rel,
            "weight": torch.tensor(w, dtype=torch.float32),
        }
