import torch
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from bench.detectors import get_detector  # Changed from .zoo to .detectors
from bench.loader import get_loader
from bench.backbones import load_backbone
from bench.datasets import DATASET_ZOO


# --- Metrics ---
def compute_id_threshold(confidences, tpr=0.95):
    if len(confidences) == 0: return 0.0
    confidences = np.sort(confidences)
    cutoff_index = int(len(confidences) * (1 - tpr))
    return confidences[cutoff_index]


def get_outcome(is_correct, confidence, threshold):
    is_accepted = confidence >= threshold
    if is_correct and is_accepted: return "Clean_Success"
    if is_correct and not is_accepted: return "Nuisance_Novelty"
    if not is_correct and not is_accepted: return "Double_Failure"
    if not is_correct and is_accepted: return "Contained_Misidentification"
    return "Error"


# bench/run_bench.py

def run_inference(model, postprocessor, loader, device, desc=""):
    results = []
    needs_grad = getattr(postprocessor, "requires_grad", False)

    print(f"Running Inference: {desc}")
    grad_ctx = torch.enable_grad() if needs_grad else torch.no_grad()

    with grad_ctx:
        for batch in tqdm(loader, desc=desc):
            if batch is None:
                continue

            img = batch["data"].to(device)
            labels = batch["label"].to(device)

            # --- RAW CLASSIFIER (ground-truth top-1 accuracy should come from here)
            logits = model(img)
            raw_preds = logits.argmax(dim=1)

            # --- DETECTOR SCORE (may also return its own preds; do NOT use for raw accuracy)
            det_preds, det_confs = postprocessor.postprocess(model, img)

            raw_preds_np = raw_preds.detach().cpu().numpy()
            det_preds_np = det_preds.detach().cpu().numpy()
            det_confs_np = det_confs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            bs = len(labels_np)
            for i in range(bs):
                results.append({
                    "dataset": batch["dataset_name"][i],
                    "path": batch["path"][i],
                    "label": int(labels_np[i]),

                    # raw classifier quantities
                    "raw_prediction": int(raw_preds_np[i]),
                    "correct_cls": int(raw_preds_np[i] == labels_np[i]),  # <--- FIXED

                    # detector quantities
                    "prediction": int(det_preds_np[i]),          # keep if you want to inspect
                    "confidence": float(det_confs_np[i]),        # detector score (direction varies!)
                    "level": int(batch["level"][i].item()),
                    "nuisance": batch["nuisance"][i],
                })

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbones', nargs='+', default=['vit_b_16', 'resnet50'])
    parser.add_argument('--detectors', nargs='+', default=['ebo', 'msp', 'react', 'ash'])
    parser.add_argument('--id_dataset', default="ImageNet-Test")
    parser.add_argument('--test_datasets', nargs='+', default=['LN_v3', 'ImageNet-C', 'LN_v2'])
    parser.add_argument('--out_dir', default="analysis/bench_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading ID Dataset (Train) for Detector Setup...")
    # NOTE: You might want a smaller subset if Train is too huge for KNN (1.2M images)
    # But for MDS, using the full train set is standard.
    id_loader_train = get_loader("ImageNet-Train", batch_size=64)
    id_loader_val = get_loader("ImageNet-Val", batch_size=64)

    # 2. Load Test Loaders
    test_loaders = []
    for name in args.test_datasets:
        print(f"Loading Test Dataset: {name}")
        test_loaders.append(get_loader(name))

    all_rows = []

    # 3. Main Loop
    # 3. Main Loop
    for bb_name in args.backbones:
        print(f"\n=== Backbone: {bb_name} ===")
        model = load_backbone(bb_name, device)

        for det_name in args.detectors:
            print(f"--- Detector: {det_name} ---")
            detector = get_detector(det_name)

            detector.setup(model, {'train': id_loader_train, 'val': id_loader_val}, None)
            # 1. Load the COSTARR Validation Sets
            # Note: Ensure these keys exist in your datasets.py
            id_val_loader = get_loader('ImageNetV2-Val', batch_size=64)
            surrogate_loader = get_loader('OpenImage-O-Surrogate', batch_size=64)

            print(f"[{det_name}] --- OOSA Calibration---")

            # 2. Get Scores for Knowns (ImageNetV2)
            print(f" > Inference on Knowns (ImageNetV2)...")
            id_results = run_inference(model, detector, id_val_loader, device)

            id_confs = np.array([r['confidence'] for r in id_results])
            id_correct_mask = np.array([bool(r['correct_cls']) for r in id_results])

            # 3. Get Scores for Unknowns (OpenImage-O)
            print(f" > Inference on Unknowns (OpenImage-O)...")
            surr_results = run_inference(model, detector, surrogate_loader, device)
            surr_confs = np.array([r['confidence'] for r in surr_results])

            # 4. Compute Threshold
            from bench.metrics import compute_oosa_threshold

            print(f" > Optimizing Threshold...")
            oosa_threshold, val_score = compute_oosa_threshold(id_confs, surr_confs, id_correct_mask)

            print(f" > OOSA Threshold: {oosa_threshold:.4f} (Val Score: {val_score:.4f})")

            # C. Benchmark Test Datasets
            # Now we use that threshold on your target Nuisance datasets
            for loader in test_loaders:
                results = run_inference(model, detector, loader, device, desc=loader.dataset.name)
                for row in results:
                    row.update({'backbone': bb_name, 'detector': det_name, 'threshold_used': oosa_threshold})
                    row['outcome'] = get_outcome(bool(row['correct_cls']), row['confidence'], oosa_threshold)
                    all_rows.append(row)

    # 4. Save
    df = pd.DataFrame(all_rows)
    cols = ['backbone', 'detector', 'dataset', 'nuisance', 'level', 'outcome',
            'correct_cls', 'confidence', 'prediction', 'label', 'path']
    df = df[[c for c in cols if c in df.columns]]

    csv_path = os.path.join(args.out_dir, "final_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()