# make_cns_bench_imglists.py
import json
import os
from pathlib import Path

ANN_PATH = "../data/annotations/annotations_cns.json"
IMGLIST_DIR = "../data/benchmark_imglist/cns_bench"

def main():
    os.makedirs(IMGLIST_DIR, exist_ok=True)

    with open(ANN_PATH, "r") as f:
        ann = json.load(f)

    images = ann["images"]

    # 1) Single combined imglist (for OpenOOD evaluator)
    all_imglist_path = Path(IMGLIST_DIR) / "cns_bench_all.txt"
    with open(all_imglist_path, "w") as f:
        for im in images:
            rel_path = im["image"]           # e.g. painting_style/004_hammerhead/...
            label = im["class"]              # ImageNet class index (int)
            f.write(f"{rel_path} {label}\n")

    print(f"Wrote combined imglist: {all_imglist_path}")

    # 2) Optional: a little metadata file for analysis (shift, scale, class)
    #    This is not required by OpenOOD, just nice for your own scripts.
    meta_path = Path(IMGLIST_DIR) / "cns_bench_meta.tsv"
    with open(meta_path, "w") as f:
        f.write("image\tclass\tshift\tscale\tseed\n")
        for im in images:
            f.write(
                f"{im['image']}\t{im['class']}\t{im['shift']}\t{im['scale']}\t{im['seed']}\n"
            )

    print(f"Wrote metadata TSV: {meta_path}")

if __name__ == "__main__":
    main()
