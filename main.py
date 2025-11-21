# main.py
import argparse
import os
import yaml
import logging
from datetime import datetime

from imagenet_c_psycho.train_pipeline import TrainPipeline as Pipeline

import glob
import copy

def expand_ratio_runs(cfg):
    """
    Returns a list of configs, one per ratio-split directory.
    """
    base_dir = "./data/benchmark_imglist/imagenet_c_psycho/"
    ratio_dirs = sorted([
        d for d in glob.glob(base_dir + "/*")
        if os.path.isdir(d)
    ])

    expanded = []

    for rdir in ratio_dirs:
        ratio_name = os.path.basename(rdir)

        new_cfg = copy.deepcopy(cfg)

        # Replace imglist paths
        new_cfg["dataset"]["train"]["imglist_pth"] = f"{rdir}/train_imagenet_c_psycho.txt"
        new_cfg["dataset"]["val"]["imglist_pth"]   = f"{rdir}/val_imagenet_c_psycho.txt"
        new_cfg["dataset"]["test"]["imglist_pth"]  = f"{rdir}/test_imagenet_c_psycho.txt"

        # Tag experiment with ratio name
        new_cfg["mark"] = ratio_name

        expanded.append(new_cfg)

    return expanded

def setup_logging(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, "train.log")

    logger = logging.getLogger("nuisance")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to {logfile}")
    return logger


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _dot_get(d, dotted, default=""):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

def expand_exp_name(template: str, cfg: dict) -> str:
    # replace tokens like @{dataset.name} using cfg values
    out = template
    import re
    for m in re.findall(r"@{([^}]+)}", template):
        val = _dot_get(cfg, m, "")
        out = out.replace(f"@{{{m}}}", str(val))
    # strip quotes that are often embedded in openood templates
    return out.replace("'", "").replace('"', "")

def resolve_save_dir(cfg: dict) -> str:
    out_root = cfg.get("output_dir", "./results")
    exp_name = cfg.get("exp_name")
    if exp_name:
        exp_name = expand_exp_name(exp_name, cfg)
    else:
        ds = cfg.get("dataset", {}).get("name", "dataset")
        net = cfg.get("network", {}).get("name", "net")
        trn = cfg.get("trainer", {}).get("name", "trainer")
        exp_name = f"{ds}_{net}_{trn}"
    return os.path.join(out_root, exp_name)


def main():
    parser = argparse.ArgumentParser("Psychophysical ImageNet-C training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--network", default=None, help="Override network name (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # optional CLI override for backbone
    if args.network:
        cfg.setdefault("network", {})
        cfg["network"]["name"] = args.network

    save_dir = resolve_save_dir(cfg)
    logger = setup_logging(save_dir)
    logger.info("Loaded config.")
    logger.info(cfg)

    all_cfgs = expand_ratio_runs(cfg)

    for sub_cfg in all_cfgs:
        ratio = sub_cfg["mark"]
        print(f"\n==============================")
        print(f" TRAINING RATIO SPLIT: {ratio}")
        print(f"==============================\n")

        exp_name = f"{ratio}_{sub_cfg['network']['name']}"
        save_dir = os.path.join(sub_cfg["output_dir"], exp_name)

        # ✅ Skip if results already exist
        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            print(f"Skipping {ratio} — results already exist at {save_dir}")
            continue

        os.makedirs(save_dir, exist_ok=True)
        pipeline = Pipeline(sub_cfg, save_dir, logger)
        pipeline.run()



if __name__ == "__main__":
    main()
