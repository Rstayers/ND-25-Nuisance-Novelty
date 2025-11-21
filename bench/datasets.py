# psycho_bench/datasets.py

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Outcome taxonomies
# -----------------------------


def classify_id_outcome(correct: int, accept: int) -> str:
    """
    4-way ID taxonomy:

        Clean_Success
        Nuisance_Novelty
        Double_Failure
        Contained_Misidentification
    """
    if correct == 1 and accept == 1:
        return "Clean_Success"
    if correct == 1 and accept == 0:
        return "Nuisance_Novelty"
    if correct == 0 and accept == 0:
        return "Double_Failure"
    # correct == 0 and accept == 1
    return "Contained_Misidentification"


def classify_ood_outcome(accept: int) -> str:
    """
    OOD taxonomy (NINCO):

        OOD_CorrectReject  (accept == 0)
        OOD_FalseAccept    (accept == 1)
    """
    if accept == 0:
        return "OOD_CorrectReject"
    return "OOD_FalseAccept"


# -----------------------------
# Generic imglist dataset
# -----------------------------


class ImglistDataset(Dataset):
    """
    imglist lines: "<rel_path> <class_id>"
    data_dir: root for images (so full path is data_dir / rel_path)
    """

    def __init__(self, imglist_path: str, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        with open(imglist_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.rsplit(" ", 1)
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        full_path = os.path.join(self.data_dir, rel_path)
        img = Image.open(full_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, rel_path


# -----------------------------
# Path parsers
# -----------------------------


def parse_cns_shift(rel_path: str) -> str:
    """
    CNS: severity is baked into the filename, not the directory.
    We just take the "shift" (e.g., 'sand', 'cartoon') from the directory.

    Examples:
      'sand/imagenet_0001_s3.png'          -> 'sand'
      'cns/cartoon/imagenet_0001_s2.png'   -> 'cartoon'
    """
    parts = Path(rel_path).parts
    if len(parts) == 0:
        return "unknown"
    first = parts[0].lower()
    # If there's a top-level "cns" directory, use the next component as the shift.
    if first in {"cns", "cns-bench", "cns_bench"} and len(parts) > 1:
        return parts[1]
    # Otherwise, treat the first path component as the shift.
    return parts[0]


def parse_imagenet_c_corruption(rel_path: str) -> str:
    """
    Extract the corruption type from paths like:
        imagenet_c/brightness/5/img.JPEG
    or
        imagenet_c_psycho/brightness/5/img.JPEG
    """
    parts = Path(rel_path).parts
    # Strip leading 'imagenet_c' or 'imagenet_c_psycho'
    if parts and parts[0].lower() in {"imagenet_c", "imagenet_c_psycho"}:
        parts = parts[1:]
    return parts[0] if len(parts) > 0 else "unknown"


def parse_imagenet_c_severity(rel_path: str) -> float:
    """
    Extract the numeric corruption severity from paths like:
        imagenet_c/brightness/5/img.JPEG
        → 5.0
    """
    parts = Path(rel_path).parts
    # Strip leading 'imagenet_c' or 'imagenet_c_psycho'
    if parts and parts[0].lower() in {"imagenet_c", "imagenet_c_psycho"}:
        parts = parts[1:]
    if len(parts) < 2:
        return 0.0
    sev_str = parts[1]
    try:
        return float(sev_str)
    except ValueError:
        return 0.0



def parse_ninco_subset(rel_path: str) -> str:
    """
    NINCO: take the first directory in the relative path as 'ood_subset'.

    Example:
      'texture/foo.jpg' -> 'texture'
    """
    parts = Path(rel_path).parts
    if len(parts) == 0:
        return "unknown"
    return parts[0]


# -----------------------------
# Dataset spec + registry
# -----------------------------
def parse_cns_severity(rel_path: str) -> float:
    """
    Parse the CNS-Bench nuisance severity from filename patterns like:
        .../seed_0024_scale_0000.jpg
    Returns the float scale (e.g. 0.0, 0.5, 1.0, etc.)
    """
    import re
    name = Path(rel_path).name
    m = re.search(r"_scale_(\d+)", name)
    if m:
        try:
            return float(m.group(1)) / 1000.0  # 0000 -> 0.0, 0500 -> 0.5
        except ValueError:
            return 0.0
    return 0.0


@dataclass
class DatasetSpec:
    name: str                   # "cns", "imagenet_c", "ninco", ...
    role: str                   # "id_shift" (CNS, ImageNet-C) or "ood" (NINCO)
    data_dir: str               # relative to data_root
    imglist: str                # relative to data_root
    parse_tag: Optional[Callable[[str], str]] = None   # shift or subset
    parse_scale: Optional[Callable[[str], float]] = None  # severity/scale if any


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "cns": DatasetSpec(
        name="cns",
        role="id_shift",
        data_dir="images_largescale/cns_bench",
        imglist="benchmark_imglist/cns_bench/cns_bench_all.txt",
        parse_tag=parse_cns_shift,
        parse_scale=parse_cns_severity,  # instead of None
    ),
    "imagenet_c": DatasetSpec(
        name="imagenet_c",
        role="id_shift",
        data_dir="images_largescale",
        imglist="benchmark_imglist/imagenet_c_psycho/0clean_100c/test_imagenet_c_psycho.txt",
        parse_tag=parse_imagenet_c_corruption,
        parse_scale=parse_imagenet_c_severity,
    ),
    "ninco": DatasetSpec(
        name="ninco",
        role="ood",
        data_dir="images_largescale",
        imglist="benchmark_imglist/imagenet/test_ninco.txt",
        parse_tag=parse_ninco_subset,
        parse_scale=None,
    ),
    # Future datasets can be added here.
}


# -----------------------------
# Row builders (per-sample CSV)
# -----------------------------






def build_id_shift_row(
    dataset_name: str,
    det_name: str,
    backbone: str,
    tau: float,
    fpr_target: float,
    full_path: str,
    shift: str,
    scale: float,
    gt: int,
    pred: int,
    score: float,
) -> Dict[str, Any]:
    """
    Generic row builder for ID-like datasets under shift (CNS, ImageNet-C, etc.).
    Uses the 4-way ID taxonomy.
    """
    correct = int(pred == gt)
    accept = int(score >= tau)
    err = classify_id_outcome(correct, accept)

    return {
        "dataset": dataset_name,   # e.g. "cns", "imagenet_c"
        "shift": shift,            # e.g. 'sand', 'cartoon', 'gaussian_noise'
        "ood_subset": "",
        "scale": float(scale),     # e.g. 5.0 for severity, or 0.0 for CNS
        "seed": -1,
        "detector": det_name,
        "backbone": backbone,
        "fpr_target": fpr_target,
        "threshold": tau,
        "image_path": full_path,
        "class_id": gt,
        "pred_class": pred,
        "score": score,
        "correct_cls": correct,
        "accept": accept,
        "reject": int(not accept),
        "error_type": err,
        "is_ood": 0,               # conceptually ID
    }




def build_ood_row(
    dataset_name: str,
    det_name: str,
    backbone: str,
    tau: float,
    fpr_target: float,
    full_path: str,
    subset: str,
    score: float,
    pred: int,
) -> Dict[str, Any]:
    """
    Generic row builder for OOD-like datasets (e.g., NINCO).
    """
    accept = int(score >= tau)
    err = classify_ood_outcome(accept)

    return {
        "dataset": dataset_name,   # e.g. "ninco"
        "shift": "",
        "ood_subset": subset,      # e.g. 'texture', 'object', ...
        "scale": -1.0,
        "seed": -1,
        "detector": det_name,
        "backbone": backbone,
        "fpr_target": fpr_target,
        "threshold": tau,
        "image_path": full_path,
        "class_id": -1,
        "pred_class": pred,
        "score": score,
        "correct_cls": 0,          # not meaningful for OOD
        "accept": accept,
        "reject": int(not accept),
        "error_type": err,
        "is_ood": 1,
    }


# -----------------------------
# Loader helper
# -----------------------------


def build_loader_for_spec(
    spec: DatasetSpec,
    data_root: str,
    batch_size: int,
    num_workers: int,
    transform=None,
) -> DataLoader:
    data_dir = os.path.join(data_root, spec.data_dir)
    imglist_path = os.path.join(data_root, spec.imglist)
    ds = ImglistDataset(imglist_path=imglist_path, data_dir=data_dir, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
