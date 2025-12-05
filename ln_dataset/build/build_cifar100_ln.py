import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from LN.config import get_severity_levels, MAX_SAMPLING_ATTEMPTS, SeverityLevel
from LN.nuisances.sticker import StickerNuisance
from LN.core.metadata import MetadataWriter


def _iter_cifar_images(
    root: Path,
    split: str,
    fraction: float,
    imglist: Optional[Path] = None,
) -> List[Path]:
    """
    Return a list of image paths for CIFAR-100-LN.

    Two modes:
      1) Directory mode (imglist is None): walk <root>/<split> and collect images.
      2) Imgalist mode (imglist is provided): read <rel_path> [label] from the
         imagelist and map to full paths under <root>.

    NOTE: Do NOT point 'split' at the official CIFAR-100 test set if you
    are generating LN images for *training*. Use a held-out portion of
    the train split (e.g. 'train_ln_base') to avoid train–test leakage.
    """
    if imglist is not None:
        # imagelist lines: "<rel_path> <label>" or just "<rel_path>"
        rel_paths: List[Path] = []
        with imglist.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_str = parts[0]
                rel_paths.append(Path(rel_str))

        all_paths = [root / rel for rel in rel_paths]
    else:
        split_root = root / split
        exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
        all_paths = [p for p in split_root.rglob("*") if p.suffix in exts]

    if fraction < 1.0 and len(all_paths) > 0:
        n = max(1, int(len(all_paths) * fraction))
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(all_paths), size=n, replace=False)
        all_paths = [all_paths[i] for i in idxs]

    return sorted(all_paths)


def build_cifar100_sticker_ln(
    cifar_root: Path,
    split: str,
    output_root: Path,
    assets_dir: Path,
    severities: List[int],
    fraction: float,
    seed: int,
    imglist: Optional[Path] = None,
) -> None:
    """
    Build CIFAR-100-LN sticker dataset from a given split of a CIFAR-100
    ImageFolder root.

    Args:
        cifar_root: root with subdirs (e.g. 'train_ln_base', 'test').
        split: which subdir to process under cifar_root (used as a tag in metadata).
        output_root: where to write stickers/{severity}/... and metadata/.
        assets_dir: directory containing stickers.yaml and PNG assets.
        severities: list of severities to generate (e.g. [1,2,3,4,5]).
        fraction: fraction of images in the split to process (0 < f <= 1).
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)

    # Base schedule from ImageNet-LN config
    base_schedule = get_severity_levels(severities)

    # Scale sticker areas down for 32x32 CIFAR-100 images.
    # ImageNet-LN areas are tuned for ~224x224; here we reduce total
    # area by a factor so stickers aren't absurdly large / impossible
    # to place under the central-coverage and max-coverage constraints.
    AREA_SCALE = 0.25  # you can tweak this if stickers are too small/large

    schedule = {
        k: SeverityLevel(
            severity=v.severity,
            n_min=v.n_min,
            n_max=v.n_max,
            area_min=v.area_min * AREA_SCALE,
            area_max=v.area_max * AREA_SCALE,
            center_min=v.center_min,
            center_max=v.center_max,
        )
        for k, v in base_schedule.items()
    }

    nuisance = StickerNuisance(assets_dir=assets_dir, schedule=schedule)

    images = _iter_cifar_images(cifar_root, split, fraction, imglist=imglist)

    meta_path = output_root / "metadata" / f"stickers_cifar100_v0.1_{split}.jsonl"
    with MetadataWriter(meta_path) as mw:
        for img_path in tqdm(images, desc=f"Building CIFAR-100-LN ({split})"):
            # orig_relpath is relative to cifar_root
            try:
                rel = img_path.relative_to(cifar_root)
            except ValueError:
                # If paths are not strictly under cifar_root, skip them
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:

                continue

            for severity in severities:
                success = False
                for _ in range(MAX_SAMPLING_ATTEMPTS):
                    try:
                        out, meta = nuisance.apply(img, severity=severity, rng=rng)
                    except RuntimeError:
                        # sampling failed under geometric/coverage constraints
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
        description="Build CIFAR-100-LN Sticker dataset (ImageNet-LN-style)."
    )
    parser.add_argument(
        "--cifar-root",
        type=str,
        required=True,
        help=(
            "Path to CIFAR-100 root with ImageFolder-style layout, i.e. "
            "the prefix for the <rel_path> entries in the imagelist "
            "(e.g. if imagelist has 'train/class_x/img.png', use the "
            "directory that contains 'train/')."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help=(
            "Logical split tag (e.g. 'train_ln_base' for training aux, "
            "'test' for LN test). Used in metadata filename; when "
            "--imglist is provided it does not need to correspond to "
            "an actual subdirectory."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root directory for CIFAR-100-LN.",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        required=True,
        help="Directory containing stickers.yaml and PNG assets.",
    )
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of severities to generate.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of images to process (0 < f <= 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--imglist",
        type=str,
        default=None,
        help=(
            "Optional imagelist file specifying which images to process "
            "(lines: '<rel_path> <label>'). If provided, overrides "
            "directory walk under --cifar-root/--split."
        ),
    )

    args = parser.parse_args()

    build_cifar100_sticker_ln(
        cifar_root=Path(args.cifar_root),
        split=args.split,
        output_root=Path(args.output_root),
        assets_dir=Path(args.assets_dir),
        severities=args.severities,
        fraction=args.fraction,
        seed=args.seed,
        imglist=Path(args.imglist) if args.imglist is not None else None,
    )


if __name__ == "__main__":
    from pathlib import Path


    main()
