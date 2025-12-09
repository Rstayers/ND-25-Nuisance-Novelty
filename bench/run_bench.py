import argparse
import os
import csv
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

# OpenOOD imports
from openood.postprocessors import get_postprocessor
from openood.evaluators.metrics import compute_all_metrics

# Local imports
from bench.backbones import load_backbone
from bench.datasets import get_dataloader, parse_nuisance_metadata
from bench.metrics import compute_thresholds, classify_outcome

# TPR Targets: 0.95 means we want to preserve 95% of ID samples.
TPR_TARGETS = [0.95, 0.99]


def run_inference(model, postprocessor, loader, device):
    """
    Raw inference loop. Returns (preds, confs, labels, paths).
    """
    preds, confs, labels, paths = [], [], [], []

    with torch.no_grad():
        for imgs, lbls, batch_paths in tqdm(loader, leave=False):
            imgs = imgs.to(device)

            # OpenOOD Postprocessor interaction
            # Note: OpenOOD postprocessors usually expect (net, input)
            _, batch_conf = postprocessor.postprocess(model, imgs)

            # Get raw predictions for accuracy check
            logits = model(imgs)
            batch_pred = logits.argmax(1)

            preds.append(batch_pred.cpu().numpy())
            confs.append(batch_conf.cpu().numpy())
            labels.append(lbls.numpy())
            paths.extend(batch_paths)

    return (np.concatenate(preds),
            np.concatenate(confs),
            np.concatenate(labels),
            paths)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Model & Postprocessor
    model = load_backbone(args.backbone, device)

    # OpenOOD config wrapper for postprocessor
    # We create a dummy config object if OpenOOD requires it,
    # but standard postprocessors (msp, ebo) usually just need args.
    class DummyConfig:
        def __init__(self):
            self.postprocessor = argparse.Namespace()
            self.postprocessor.name = args.detector
            self.postprocessor.APS_mode = False  # Add args as needed

    config = DummyConfig()
    postprocessor = get_postprocessor(config, "id_name_dummy")
    postprocessor.setup(model, None, None)  # Setup if needed

    # 2. Process ID Data (ImageNet Validation)
    print(f"--- Processing ID: {args.id_name} ---")
    id_loader = get_dataloader(args.data_root, args.id_imglist, args.batch_size, args.num_workers,
                               postprocessor.transform)
    id_pred, id_conf, id_gt, _ = run_inference(model, postprocessor, id_loader, device)

    # 3. Compute Thresholds (THE CRITICAL FIX)
    # Thresholds are locked here based purely on ID statistics.
    thresholds = compute_thresholds(id_conf, TPR_TARGETS)
    print(f"Calculated Thresholds (ID={args.id_name}): {thresholds}")

    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # 4. Process OOD/Nuisance Datasets
    # We stream results to a single CSV to save memory.
    csv_path = os.path.join(args.out_dir, f"results_{args.backbone}_{args.detector}.csv")

    fieldnames = ['dataset', 'nuisance', 'severity', 'path', 'label', 'pred', 'conf', 'correct']
    for tpr in TPR_TARGETS:
        fieldnames.append(f"outcome_tpr{tpr}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # ID Entries (Optional, but good for completeness)
        # ... logic to write ID entries if desired ...

        for ds_name, imglist in zip(args.ood_names, args.ood_imglists):
            print(f"--- Processing OOD: {ds_name} ---")
            loader = get_dataloader(args.data_root, imglist, args.batch_size, args.num_workers, postprocessor.transform)
            ood_pred, ood_conf, ood_gt, ood_paths = run_inference(model, postprocessor, loader, device)

            # OpenOOD Standard Metrics (AUROC, FPR95)
            # We treat this dataset as strictly OOD (-1 labels) for standard metric calculation
            # This prints the console summary
            print(f"Standard Metrics for {ds_name}:")
            metrics = compute_all_metrics(
                np.concatenate([id_conf, ood_conf]),
                np.concatenate([np.zeros_like(id_conf), np.ones_like(ood_conf)]),  # 0=ID, 1=OOD
                np.concatenate([id_pred, ood_pred])
            )
            # Note: compute_all_metrics prints automatically or returns dict.
            # If it returns tuple, handle accordingly. OpenOOD changes versions often.

            # Per-Sample Analysis & Writing
            for i in range(len(ood_conf)):
                rel_path = ood_paths[i]
                nuisance, severity = parse_nuisance_metadata(rel_path)
                is_correct = (ood_pred[i] == ood_gt[i])

                row = {
                    'dataset': ds_name,
                    'nuisance': nuisance,
                    'severity': severity,
                    'path': rel_path,
                    'label': ood_gt[i],
                    'pred': ood_pred[i],
                    'conf': f"{ood_conf[i]:.5f}",
                    'correct': int(is_correct)
                }

                for tpr in TPR_TARGETS:
                    outcome = classify_outcome(
                        is_id=False,
                        is_correct=is_correct,
                        score=ood_conf[i],
                        threshold=thresholds[tpr]
                    )
                    row[f"outcome_tpr{tpr}"] = outcome

                writer.writerow(row)

    print(f"Benchmark Complete. Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Base path for images")

    # ID Args
    parser.add_argument("--id_name", type=str, default="imagenet")
    parser.add_argument("--id_imglist", type=str, required=True)

    # OOD Args (Lists)
    parser.add_argument("--ood_names", nargs='+', required=True, help="List of dataset names")
    parser.add_argument("--ood_imglists", nargs='+', required=True, help="List of imglist paths matching names")

    # Model Args
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--detector", type=str, default="msp")

    parser.add_argument("--out_dir", type=str, default="results/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    main(args)