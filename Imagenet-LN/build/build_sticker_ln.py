import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import get_severity_levels, MAX_SAMPLING_ATTEMPTS
from ..nuisances.sticker import StickerNuisance
from ..core.metadata import MetadataWriter


def _iter_imagenet_images(root: Path, split: str, fraction: float) -> List[Path]:
    """
    Recursively list all image files under root/split.
    Supports either flat val/ or class-subdir structure.
    """
    split_root = root / split
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    all_paths = [p for p in split_root.rglob("*") if p.suffix in exts]
    if fraction < 1.0:
        n = max(1, int(len(all_paths) * fraction))
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(all_paths), size=n, replace=False)
        all_paths = [all_paths[i] for i in idxs]
    return sorted(all_paths)


def build_sticker_ln(
    imagenet_root: Path,
    split: str,
    output_root: Path,
    assets_dir: Path,
    severities: List[int],
    fraction: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    schedule = get_severity_levels(severities)
    nuisance = StickerNuisance(assets_dir=assets_dir, schedule=schedule)

    images = _iter_imagenet_images(imagenet_root, split, fraction)
    meta_path = output_root / "metadata" / f"stickers_v0.1_{split}.jsonl"
    with MetadataWriter(meta_path) as mw:

        for img_path in tqdm(images, desc=f"Building ImageNet-LN ({split})"):
            rel = img_path.relative_to(imagenet_root / split)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            for severity in severities:
                # sample with limited retries to avoid pathological cases
                success = False
                for _ in range(MAX_SAMPLING_ATTEMPTS):
                    try:
                        out, meta = nuisance.apply(img, severity=severity, rng=rng)
                    except RuntimeError:
                        # sampling failed for constraints; skip if persistent
                        continue
                    success = True
                    break

                if not success:
                    continue

                out_rel = Path(f"stickers/{severity}") / rel
                out_path = output_root / out_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out.save(out_path.with_suffix(".png"))

                record = {
                    "orig_relpath": str(rel),
                    "ln_relpath": str(out_rel.with_suffix(".png")),
                    "severity": severity,
                    "seed": int(rng.integers(0, 2**31 - 1)),
                    **meta,
                }
                mw.write(record)


def main():
    parser = argparse.ArgumentParser(
        description="Build ImageNet-LN Sticker MVP dataset."
    )
    parser.add_argument(
        "--imagenet-root", type=str, required=True,
        help="Path to ImageNet root with 'train' and/or 'val' subdirs.",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val"],
        help="Which split to process.",
    )
    parser.add_argument(
        "--output-root", type=str, required=True,
        help="Output root directory for ImageNet-LN.",
    )
    parser.add_argument(
        "--assets-dir", type=str, required=True,
        help="Directory containing stickers.yaml and PNG assets.",
    )
    parser.add_argument(
        "--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="List of severities to generate.",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of images to process (0 < f <= 1).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    build_sticker_ln(
        imagenet_root=Path(args.imagenet_root),
        split=args.split,
        output_root=Path(args.output_root),
        assets_dir=Path(args.assets_dir),
        severities=args.severities,
        fraction=args.fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import get_severity_levels, MAX_SAMPLING_ATTEMPTS
from ..nuisances.sticker import StickerNuisance
from ..core.metadata import MetadataWriter


def _iter_imagenet_images(root: Path, split: str, fraction: float) -> List[Path]:
    """
    Recursively list all image files under root/split.
    Supports either flat val/ or class-subdir structure.
    """
    split_root = root / split
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    all_paths = [p for p in split_root.rglob("*") if p.suffix in exts]
    if fraction < 1.0:
        n = max(1, int(len(all_paths) * fraction))
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(all_paths), size=n, replace=False)
        all_paths = [all_paths[i] for i in idxs]
    return sorted(all_paths)


def build_sticker_ln(
    imagenet_root: Path,
    split: str,
    output_root: Path,
    assets_dir: Path,
    severities: List[int],
    fraction: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    schedule = get_severity_levels(severities)
    nuisance = StickerNuisance(assets_dir=assets_dir, schedule=schedule)

    images = _iter_imagenet_images(imagenet_root, split, fraction)
    meta_path = output_root / "metadata" / f"stickers_v0.1_{split}.jsonl"
    with MetadataWriter(meta_path) as mw:

        for img_path in tqdm(images, desc=f"Building ImageNet-LN ({split})"):
            rel = img_path.relative_to(imagenet_root / split)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            for severity in severities:
                # sample with limited retries to avoid pathological cases
                success = False
                for _ in range(MAX_SAMPLING_ATTEMPTS):
                    try:
                        out, meta = nuisance.apply(img, severity=severity, rng=rng)
                    except RuntimeError:
                        # sampling failed for constraints; skip if persistent
                        continue
                    success = True
                    break

                if not success:
                    continue

                out_rel = Path(f"stickers/{severity}") / rel
                out_path = output_root / out_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out.save(out_path.with_suffix(".png"))

                record = {
                    "orig_relpath": str(rel),
                    "ln_relpath": str(out_rel.with_suffix(".png")),
                    "severity": severity,
                    "seed": int(rng.integers(0, 2**31 - 1)),
                    **meta,
                }
                mw.write(record)


def main():
    parser = argparse.ArgumentParser(
        description="Build ImageNet-LN Sticker MVP dataset."
    )
    parser.add_argument(
        "--imagenet-root", type=str, required=True,
        help="Path to ImageNet root with 'train' and/or 'val' subdirs.",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val"],
        help="Which split to process.",
    )
    parser.add_argument(
        "--output-root", type=str, required=True,
        help="Output root directory for ImageNet-LN.",
    )
    parser.add_argument(
        "--assets-dir", type=str, required=True,
        help="Directory containing stickers.yaml and PNG assets.",
    )
    parser.add_argument(
        "--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="List of severities to generate.",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of images to process (0 < f <= 1).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    build_sticker_ln(
        imagenet_root=Path(args.imagenet_root),
        split=args.split,
        output_root=Path(args.output_root),
        assets_dir=Path(args.assets_dir),
        severities=args.severities,
        fraction=args.fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
