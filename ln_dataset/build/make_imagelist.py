import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import csv

# --------------------------------------------------------
#  Helper loaders
# --------------------------------------------------------

def _iter_metadata_lines(metadata_files: List[Path]) -> Iterable[Dict]:
    """Yield metadata JSON objects from one or more metadata JSONL files."""
    for mf in metadata_files:
        mf = mf.expanduser().resolve()
        if not mf.is_file():
            raise FileNotFoundError(f"Metadata file not found: {mf}")
        with mf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
def _load_label_map_from_cifar_imagelist(imagelist_path: Path) -> Dict[str, int]:
    """
    Load mapping from *full relative path* (e.g. 'cifar100/test/lamp/0037.png')
    to class index, using an existing CIFAR imagelist file:
        <relpath> <label>
    """
    imagelist_path = imagelist_path.expanduser().resolve()
    if not imagelist_path.is_file():
        raise FileNotFoundError(f"Label-source imagelist not found: {imagelist_path}")

    label_map: Dict[str, int] = {}
    with imagelist_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            relpath, label_str = line.split()
            label = int(label_str)
            # Key is the full relative path string, matching 'orig_relpath' in CIFAR-LN metadata
            label_map[relpath] = label

    if not label_map:
        raise ValueError(f"No entries loaded from {imagelist_path}")
    return label_map

def _load_label_map_from_csv_synset(csv_path: Path, synset_map: Dict[str, int]) -> Dict[str, int]:
    """
    Load mapping from ImageId (and related forms) to class index, using a CSV
    with columns: ImageId, PredictionString.

    PredictionString starts with a synset, e.g.:
        n01978287 240 170 260 240 ...

    We use the FIRST synset token and synset_map to get a label index.
    """
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV label file not found: {csv_path}")

    label_map: Dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "ImageId" not in reader.fieldnames or "PredictionString" not in reader.fieldnames:
            raise ValueError("CSV must contain 'ImageId' and 'PredictionString' columns.")

        for row in reader:
            img_id = row["ImageId"].strip()  # e.g. 'ILSVRC2012_val_00048981' or 'n02017213_7894'
            pred_str = row["PredictionString"].strip()
            if not img_id or not pred_str:
                continue

            tokens = pred_str.split()
            if not tokens:
                continue
            syn = tokens[0]  # first token is synset, e.g. 'n01978287'
            if syn not in synset_map:
                # synset not in canonical list; skip
                continue
            label = synset_map[syn]

            # Store mapping on reasonable keys:
            #  - bare ImageId
            #  - with common extensions (JPEG/JPG), since orig_relpath may include extension
            label_map[img_id] = label
            label_map[img_id + ".JPEG"] = label
            label_map[img_id + ".JPG"] = label
            label_map[img_id + ".jpeg"] = label
            label_map[img_id + ".jpg"] = label

    if not label_map:
        raise ValueError(f"No labels loaded from CSV {csv_path}")
    return label_map

def _load_label_map_from_imagelist(imagelist_path: Path) -> Dict[str, int]:
    """
    Load mapping from *filename* (e.g. 'ILSVRC2012_val_00048204.JPEG')
    to class index, using an existing ImageNet imagelist file:
        <path> <label>
    """
    imagelist_path = imagelist_path.expanduser().resolve()
    if not imagelist_path.is_file():
        raise FileNotFoundError(f"Label-source imagelist not found: {imagelist_path}")

    label_map: Dict[str, int] = {}
    with imagelist_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            relpath, label_str = line.split()
            label = int(label_str)
            basename = Path(relpath).name
            label_map[basename] = label
    if not label_map:
        raise ValueError(f"No entries loaded from {imagelist_path}")
    return label_map


def _load_synset_to_idx(synset_file: Path) -> Dict[str, int]:
    """
    Load synset-to-index map from a synset file.
    Each line: 'n01440764 tench, Tinca tinca'
    """
    synset_file = synset_file.expanduser().resolve()
    synset_map: Dict[str, int] = {}
    with synset_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Extract only the first token (synset ID)
            syn = line.split()[0]
            synset_map[syn] = i
    if not synset_map:
        raise ValueError(f"No synsets loaded from {synset_file}")
    return synset_map


# --------------------------------------------------------
#  Core function
# --------------------------------------------------------

def make_imagelist(
    metadata_files: List[Path],
    output_path: Path,
    dataset_prefix: str,
    label_mode: str = "constant",
    constant_label: int = 0,
    label_source: Optional[Path] = None,
    synset_file: Optional[Path] = None,
    csv_labels: Optional[Path] = None,
) -> None:

    """
    Generate an OpenOOD-style imagelist file from Imagenet-LN metadata.

    Each line: "<relative_path> <gt_label>"

    label_mode options:
      - 'constant': all samples get constant_label
      - 'imagenet': infer label by matching basename with an ImageNet imagelist
      - 'synset': infer label from synset extracted from orig_relpath and a
                  synset->index file
    """
    label_mode = label_mode.lower()
    if label_mode not in {"constant", "imagenet", "synset"}:
        raise ValueError("label_mode must be one of: constant, imagenet, synset")

    label_map: Dict[str, int] = {}
    synset_map: Dict[str, int] = {}
    synsets: List[str] = []

    if label_mode == "imagenet":
        if label_source is None:
            raise ValueError("--label-source is required for label-mode=imagenet")
        label_map = _load_label_map_from_imagelist(label_source)

    elif label_mode == "synset":
        if synset_file is None:
            raise ValueError("--synset-file is required for label-mode=synset")
        if csv_labels is None:
            raise ValueError("--csv-labels is required for label-mode=synset")
        # canonical synset -> index
        synset_map = _load_synset_to_idx(synset_file)
        synsets = list(synset_map.keys())
        # image-id -> label (via synset)
        label_map = _load_label_map_from_csv_synset(csv_labels, synset_map)

    dataset_prefix = dataset_prefix.strip().strip("/")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for rec in _iter_metadata_lines(metadata_files):
            ln_rel = rec.get("ln_relpath")
            if not ln_rel:
                continue

            ln_rel_posix = Path(ln_rel).as_posix()
            rel_path = f"{dataset_prefix}/{ln_rel_posix}" if dataset_prefix else ln_rel_posix

            # Decide label
            if label_mode == "constant":
                label = constant_label
            elif label_mode == "imagenet":
                basename = Path(rec.get("orig_relpath", "")).name
                label = label_map.get(basename)
                if label is None:
                    continue
            elif label_mode == "synset":
                orig_rel = rec.get("orig_relpath", "")
                # We try stem first (e.g., 'ILSVRC2012_val_00048981'), then full name
                path_obj = Path(orig_rel)
                stem = path_obj.stem
                name = path_obj.name

                label = None
                if stem in label_map:
                    label = label_map[stem]
                elif name in label_map:
                    label = label_map[name]

                if label is None:
                    # No mapping for this image; skip
                    continue

                # Optional sanity check: ensure label index is in range
                if synsets and not (0 <= label < len(synsets)):
                    continue

            out_f.write(f"{rel_path} {label}\n")
            n_written += 1

    print(f"Wrote {n_written} entries to {output_path}")


# --------------------------------------------------------
#  CLI
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an OpenOOD-style imagelist from Imagenet-LN metadata."
    )
    parser.add_argument("--metadata", type=str, nargs="+", required=True,
                        help="One or more metadata JSONL files.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output imagelist .txt path.")
    parser.add_argument("--dataset-prefix", type=str, default="imagenet_ln",
                        help="Prefix before ln_relpath in imagelist paths.")
    parser.add_argument("--label-mode", type=str, default="constant",
                        choices=["constant", "imagenet", "synset"],
                        help="Label mode: constant / imagenet / synset.")
    parser.add_argument("--constant-label", type=int, default=0,
                        help="Label for constant mode.")
    parser.add_argument("--label-source", type=str, default=None,
                        help="Existing ImageNet imagelist for label-mode=imagenet.")
    parser.add_argument("--synset-file", type=str, default=None,
                        help="Text file mapping synsets to indices for label-mode=synset.")
    parser.add_argument(
        "--csv-labels",
        type=str,
        default=None,
        help="CSV file with columns ImageId,PredictionString for label-mode=synset.",
    )

    args = parser.parse_args()

    metadata_files = [Path(p) for p in args.metadata]
    label_source = Path(args.label_source) if args.label_source else None
    synset_file = Path(args.synset_file) if args.synset_file else None
    csv_labels = Path(args.csv_labels) if args.csv_labels else None

    make_imagelist(
        metadata_files=metadata_files,
        output_path=Path(args.output),
        dataset_prefix=args.dataset_prefix,
        label_mode=args.label_mode,
        constant_label=args.constant_label,
        label_source=label_source,
        synset_file=synset_file,
        csv_labels=csv_labels,  # <-- IMPORTANT
    )



if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import csv

# --------------------------------------------------------
#  Helper loaders
# --------------------------------------------------------

def _iter_metadata_lines(metadata_files: List[Path]) -> Iterable[Dict]:
    """Yield metadata JSON objects from one or more metadata JSONL files."""
    for mf in metadata_files:
        mf = mf.expanduser().resolve()
        if not mf.is_file():
            raise FileNotFoundError(f"Metadata file not found: {mf}")
        with mf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def _load_label_map_from_csv_synset(csv_path: Path, synset_map: Dict[str, int]) -> Dict[str, int]:
    """
    Load mapping from ImageId (and related forms) to class index, using a CSV
    with columns: ImageId, PredictionString.

    PredictionString starts with a synset, e.g.:
        n01978287 240 170 260 240 ...

    We use the FIRST synset token and synset_map to get a label index.
    """
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV label file not found: {csv_path}")

    label_map: Dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "ImageId" not in reader.fieldnames or "PredictionString" not in reader.fieldnames:
            raise ValueError("CSV must contain 'ImageId' and 'PredictionString' columns.")

        for row in reader:
            img_id = row["ImageId"].strip()  # e.g. 'ILSVRC2012_val_00048981' or 'n02017213_7894'
            pred_str = row["PredictionString"].strip()
            if not img_id or not pred_str:
                continue

            tokens = pred_str.split()
            if not tokens:
                continue
            syn = tokens[0]  # first token is synset, e.g. 'n01978287'
            if syn not in synset_map:
                # synset not in canonical list; skip
                continue
            label = synset_map[syn]

            # Store mapping on reasonable keys:
            #  - bare ImageId
            #  - with common extensions (JPEG/JPG), since orig_relpath may include extension
            label_map[img_id] = label
            label_map[img_id + ".JPEG"] = label
            label_map[img_id + ".JPG"] = label
            label_map[img_id + ".jpeg"] = label
            label_map[img_id + ".jpg"] = label

    if not label_map:
        raise ValueError(f"No labels loaded from CSV {csv_path}")
    return label_map

def _load_label_map_from_imagelist(imagelist_path: Path) -> Dict[str, int]:
    """
    Load mapping from *filename* (e.g. 'ILSVRC2012_val_00048204.JPEG')
    to class index, using an existing ImageNet imagelist file:
        <path> <label>
    """
    imagelist_path = imagelist_path.expanduser().resolve()
    if not imagelist_path.is_file():
        raise FileNotFoundError(f"Label-source imagelist not found: {imagelist_path}")

    label_map: Dict[str, int] = {}
    with imagelist_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            relpath, label_str = line.split()
            label = int(label_str)
            basename = Path(relpath).name
            label_map[basename] = label
    if not label_map:
        raise ValueError(f"No entries loaded from {imagelist_path}")
    return label_map


def _load_synset_to_idx(synset_file: Path) -> Dict[str, int]:
    """
    Load synset-to-index map from a synset file.
    Each line: 'n01440764 tench, Tinca tinca'
    """
    synset_file = synset_file.expanduser().resolve()
    synset_map: Dict[str, int] = {}
    with synset_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Extract only the first token (synset ID)
            syn = line.split()[0]
            synset_map[syn] = i
    if not synset_map:
        raise ValueError(f"No synsets loaded from {synset_file}")
    return synset_map


# --------------------------------------------------------
#  Core function
# --------------------------------------------------------

def make_imagelist(
    metadata_files: List[Path],
    output_path: Path,
    dataset_prefix: str,
    label_mode: str = "constant",
    constant_label: int = 0,
    label_source: Optional[Path] = None,
    synset_file: Optional[Path] = None,
    csv_labels: Optional[Path] = None,
) -> None:

    """
    Generate an OpenOOD-style imagelist file from Imagenet-LN metadata.

    Each line: "<relative_path> <gt_label>"

    label_mode options:
      - 'constant': all samples get constant_label
      - 'imagenet': infer label by matching basename with an ImageNet imagelist
      - 'synset': infer label from synset extracted from orig_relpath and a
                  synset->index file
    """
    label_mode = label_mode.lower()
    if label_mode not in {"constant", "imagenet", "synset", "cifar100"}:
        raise ValueError("label_mode must be one of: constant, imagenet, synset, cifar100")

    label_map: Dict[str, int] = {}
    synset_map: Dict[str, int] = {}
    synsets: List[str] = []

    if label_mode == "imagenet":
        if label_source is None:
            raise ValueError("--label-source is required for label-mode=imagenet")
        label_map = _load_label_map_from_imagelist(label_source)

    elif label_mode == "synset":
        if synset_file is None:
            raise ValueError("--synset-file is required for label-mode=synset")
        if csv_labels is None:
            raise ValueError("--csv-labels is required for label-mode=synset")
        # canonical synset -> index
        synset_map = _load_synset_to_idx(synset_file)
        synsets = list(synset_map.keys())
        # image-id -> label (via synset)
        label_map = _load_label_map_from_csv_synset(csv_labels, synset_map)
    elif label_mode == "cifar100":
        if label_source is None:
            raise ValueError("--label-source is required for label-mode=cifar100")
        label_map = _load_label_map_from_cifar_imagelist(label_source)
    dataset_prefix = dataset_prefix.strip().strip("/")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for rec in _iter_metadata_lines(metadata_files):
            ln_rel = rec.get("ln_relpath")
            if not ln_rel:
                continue

            ln_rel_posix = Path(ln_rel).as_posix()
            rel_path = f"{dataset_prefix}/{ln_rel_posix}" if dataset_prefix else ln_rel_posix

            # Decide label
            if label_mode == "constant":
                label = constant_label
            elif label_mode == "imagenet":
                basename = Path(rec.get("orig_relpath", "")).name
                label = label_map.get(basename)
                if label is None:
                    continue
            elif label_mode == "synset":
                orig_rel = rec.get("orig_relpath", "")
                # We try stem first (e.g., 'ILSVRC2012_val_00048981'), then full name
                path_obj = Path(orig_rel)
                stem = path_obj.stem
                name = path_obj.name

                label = None
                if stem in label_map:
                    label = label_map[stem]
                elif name in label_map:
                    label = label_map[name]

                if label is None:
                    # No mapping for this image; skip
                    continue

                # Optional sanity check: ensure label index is in range
                if synsets and not (0 <= label < len(synsets)):
                    continue
            elif label_mode == "cifar100":
                # Use full orig_relpath as key, to avoid basename collisions
                orig_rel = rec.get("orig_relpath", "")
                label = label_map.get(orig_rel)
                if label is None:
                    # no mapping for this image; skip
                    continue
            out_f.write(f"{rel_path} {label}\n")
            n_written += 1

    print(f"Wrote {n_written} entries to {output_path}")


# --------------------------------------------------------
#  CLI
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an OpenOOD-style imagelist from Imagenet-LN metadata."
    )
    parser.add_argument("--metadata", type=str, nargs="+", required=True,
                        help="One or more metadata JSONL files.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output imagelist .txt path.")
    parser.add_argument("--dataset-prefix", type=str, default="imagenet_ln",
                        help="Prefix before ln_relpath in imagelist paths.")
    parser.add_argument("--label-mode", type=str, default="constant",
                        choices=["constant", "imagenet", "synset", "cifar100"],
                        help="Label mode: constant / imagenet / synset.")
    parser.add_argument("--constant-label", type=int, default=0,
                        help="Label for constant mode.")
    parser.add_argument("--label-source", type=str, default=None,
                        help="Existing ImageNet imagelist for label-mode=imagenet.")
    parser.add_argument("--synset-file", type=str, default=None,
                        help="Text file mapping synsets to indices for label-mode=synset.")
    parser.add_argument(
        "--csv-labels",
        type=str,
        default=None,
        help="CSV file with columns ImageId,PredictionString for label-mode=synset.",
    )

    args = parser.parse_args()

    metadata_files = [Path(p) for p in args.metadata]
    label_source = Path(args.label_source) if args.label_source else None
    synset_file = Path(args.synset_file) if args.synset_file else None
    csv_labels = Path(args.csv_labels) if args.csv_labels else None

    make_imagelist(
        metadata_files=metadata_files,
        output_path=Path(args.output),
        dataset_prefix=args.dataset_prefix,
        label_mode=args.label_mode,
        constant_label=args.constant_label,
        label_source=label_source,
        synset_file=synset_file,
        csv_labels=csv_labels,  # <-- IMPORTANT
    )



if __name__ == "__main__":
    main()
