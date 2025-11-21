#!/usr/bin/env python3
import os
import random
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DATA_ROOT = Path("data")

IMGNET1K_IMGLIST = DATA_ROOT / "benchmark_imglist" / "imagenet" / "train_imagenet.txt"
IMAGENETC_IMGLIST = DATA_ROOT / "benchmark_imglist" / "imagenet" / "test_imagenet_c.txt"

BASE_OUT = DATA_ROOT / "benchmark_imglist" / "imagenet_c_psycho"
BASE_OUT.mkdir(parents=True, exist_ok=True)
TRAIN_PROP = 0.80
VAL_PROP   = 0.10
TEST_PROP  = 0.10

RATIO_PRESETS = {
    "50clean_50c": {
        "train": {"clean_ratio": 0.50, "cor_ratio": 0.50, "severities": [1, 2, 3, 4, 5]},
        "val":   {"clean_ratio": 0.50, "cor_ratio": 0.50, "severities": [1, 2, 3, 4, 5]},
        # test cfg is ignored for now; test is always 100% IN-C
        "test":  {"clean_ratio": 0.00, "cor_ratio": 1.00, "severities": [1, 2, 3, 4, 5]},
    },
    "60clean_40c": {
        "train": {"clean_ratio": 0.60, "cor_ratio": 0.40, "severities": [1, 2, 3, 4, 5]},
        "val":   {"clean_ratio": 0.60, "cor_ratio": 0.40, "severities": [1, 2, 3, 4, 5]},
        "test":  {"clean_ratio": 0.00, "cor_ratio": 1.00, "severities": [1, 2, 3, 4, 5]},
    },

    "0clean_100c": {
        "train": {"clean_ratio": 0.0, "cor_ratio": 1.00, "severities": [1, 2, 3, 4, 5]},
        "val":   {"clean_ratio": 0.0, "cor_ratio": 1.00, "severities": [1, 2, 3, 4, 5]},
        "test":  {"clean_ratio": 0.00, "cor_ratio": 1.00, "severities": [1, 2, 3, 4, 5]},
    },
}


def read_imglist(path: Path):
    xs = []
    with path.open("r") as f:
        for line in f:
            rel, lab = line.strip().rsplit(" ", 1)
            xs.append((rel, int(lab)))
    return xs


def gather_imagenet_c(path: Path):
    """
    Parse ImageNet-C imglist.
    Returns dict: (corruption, severity) -> [(rel, lab), ...]
    """
    buckets = defaultdict(list)

    with path.open("r") as f:
        for line in f:
            rel, lab = line.strip().rsplit(" ", 1)
            lab = int(lab)

            parts = rel.split("/")
            # ["imagenet_c", "<corruption>", "<severity>", "ILSVRC2012_val_XXXXX.JPEG"]
            corruption = parts[1]
            severity = int(parts[2])

            buckets[(corruption, severity)].append((rel, lab))

    return buckets


def split_with_proportions(items, train_prop, val_prop, test_prop):
    """Split list of items deterministically using proportions."""
    items = list(items)  # avoid in-place shuffling of caller's list
    random.shuffle(items)

    N = len(items)
    train_n = int(N * train_prop)
    val_n   = int(N * val_prop)
    test_n  = N - train_n - val_n

    train = items[:train_n]
    val   = items[train_n:train_n+val_n]
    test  = items[train_n+val_n:]
    return train, val, test


def uniform_sample_from_bucket(bucket, k):
    """Sample k items from a list without replacement."""
    if k > len(bucket):
        raise RuntimeError("Not enough items in bucket for uniform sampling.")
    return random.sample(bucket, k)


def write_list(path, name, items):
    out = path / f"{name}_imagenet_c_psycho.txt"
    with out.open("w") as f:
        for rel, lab in items:
            f.write(f"{rel} {lab}\n")


# ============================================================
# MASTER GENERATION LOGIC
# ============================================================

def main():
    random.seed(0)

    # clean ImageNet-1k pool
    clean_pool = read_imglist(IMGNET1K_IMGLIST)

    # corrupted ImageNet-C buckets -> flatten
    c_buckets = gather_imagenet_c(IMAGENETC_IMGLIST)
    all_corrupted = []
    for items in c_buckets.values():
        all_corrupted.extend(items)

    # ------------------------------------------------------
    # 1. Fix a single IN-C train/val/test split ONCE
    # ------------------------------------------------------
    train_cor_all, val_cor_all, test_cor_all = split_with_proportions(
        all_corrupted,
        TRAIN_PROP,
        VAL_PROP,
        TEST_PROP
    )

    print("Global corrupted split sizes:")
    print(f"  train_cor_all: {len(train_cor_all)}")
    print(f"  val_cor_all:   {len(val_cor_all)}")
    print(f"  test_cor_all:  {len(test_cor_all)}")

    # helper to build a mixed split for train/val
    def build_mixed_split(cor_subset, clean_ratio, cor_ratio):
        total_needed = len(cor_subset)
        if total_needed == 0:
            return []

        # how many corrupted vs clean we want
        cor_needed = int(total_needed * cor_ratio)
        clean_needed = total_needed - cor_needed

        # clamp in case ratios ask for more clean than exist
        clean_needed = min(clean_needed, len(clean_pool))

        cor_items = uniform_sample_from_bucket(cor_subset, cor_needed)
        clean_items = random.sample(clean_pool, clean_needed)

        merged = cor_items + clean_items
        random.shuffle(merged)
        return merged

    # ------------------------------------------------------
    # 2. Build splits per preset
    # ------------------------------------------------------
    for preset_name, preset_cfg in RATIO_PRESETS.items():
        print(f"\n==============================")
        print(f"GENERATING PRESET: {preset_name}")
        print(f"==============================")

        out_dir = BASE_OUT / preset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = preset_cfg["train"]
        val_cfg   = preset_cfg["val"]
        # test_cfg = preset_cfg["test"]  # currently unused; test is 100% IN-C

        # train/val: mix clean + corrupted according to ratios
        train = build_mixed_split(train_cor_all,
                                  train_cfg["clean_ratio"],
                                  train_cfg["cor_ratio"])
        val   = build_mixed_split(val_cor_all,
                                  val_cfg["clean_ratio"],
                                  val_cfg["cor_ratio"])

        # test: SAME across all presets, 100% corrupted, no clean images
        test = list(test_cor_all)  # shallow copy; contents identical

        # ------------------------------------------------------
        # 3. Write to disk
        # ------------------------------------------------------
        write_list(out_dir, "train", train)
        write_list(out_dir, "val",   val)
        write_list(out_dir, "test",  test)

        print(f"  Train (mixed): {len(train)}")
        print(f"  Val (mixed):   {len(val)}")
        print(f"  Test (IN-C):   {len(test)}")


if __name__ == "__main__":
    main()
