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


# --- Inference ---
def run_inference(model, postprocessor, loader, device, desc=""):
    """
    Runs inference. Does NOT call setup() anymore.
    """
    results = []

    print(f"Running Inference: {desc}")
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if batch is None: continue

            # FIX: Use 'data' key
            img = batch['data'].to(device)
            labels = batch['label'].to(device)

            # OpenOOD Call
            preds, confs = postprocessor.postprocess(model, img)

            preds = preds.cpu().numpy()
            confs = confs.cpu().numpy()
            labels = labels.cpu().numpy()

            bs = len(preds)
            for i in range(bs):
                results.append({
                    'dataset': batch['dataset_name'][i],
                    'path': batch['path'][i],
                    'label': labels[i],
                    'prediction': preds[i],
                    'confidence': confs[i],
                    'correct_cls': int(preds[i] == labels[i]),
                    'level': batch['level'][i].item(),
                    'nuisance': batch['nuisance'][i],
                    'competency_parce': batch['parce'][i].item()
                })
    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbones', nargs='+', default=['resnet50', 'vit_b_16'])
    parser.add_argument('--detectors', nargs='+', default=['msp', 'react', 'ash'])
    parser.add_argument('--id_dataset', default="ImageNet-Val")
    parser.add_argument('--test_datasets', nargs='+', required=True)
    parser.add_argument('--out_dir', default="analysis/bench_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load ID Loader (Calibration)
    print(f"Loading ID Dataset: {args.id_dataset}")
    id_loader = get_loader(args.id_dataset)

    # 2. Load Test Loaders
    test_loaders = []
    for name in args.test_datasets:
        print(f"Loading Test Dataset: {name}")
        test_loaders.append(get_loader(name))

    all_rows = []

    # 3. Main Loop
    for bb_name in args.backbones:
        print(f"\n=== Backbone: {bb_name} ===")
        model = load_backbone(bb_name, device)

        for det_name in args.detectors:
            print(f"--- Detector: {det_name} ---")
            detector = get_detector(det_name)

            # A. SETUP (Critical Fix)
            # ReAct/ASH need to see ID data to calculate statistics.
            # OpenOOD expects a dict: {'val': loader} or {'id': loader}
            print(f"[{det_name}] Calibrating statistics on ID data...")
            detector.setup(model, {'val': id_loader}, None)

            # B. Run Inference on ID (for Thresholds)
            id_results = run_inference(model, detector, id_loader, device, desc="ID Calibration")
            id_confs = [r['confidence'] for r in id_results]
            threshold = compute_id_threshold(id_confs, tpr=0.95)
            print(f" > Threshold (95% TPR): {threshold:.4f}")

            # Save ID rows
            for row in id_results:
                row.update({'backbone': bb_name, 'detector': det_name, 'threshold_used': threshold})
                row['outcome'] = get_outcome(bool(row['correct_cls']), row['confidence'], threshold)
                all_rows.append(row)

            # C. Benchmark Test Datasets
            for loader in test_loaders:
                results = run_inference(model, detector, loader, device, desc=loader.dataset.name)
                for row in results:
                    row.update({'backbone': bb_name, 'detector': det_name, 'threshold_used': threshold})
                    row['outcome'] = get_outcome(bool(row['correct_cls']), row['confidence'], threshold)
                    all_rows.append(row)

    # 4. Save
    df = pd.DataFrame(all_rows)
    cols = ['backbone', 'detector', 'dataset', 'nuisance', 'level', 'outcome',
            'correct_cls', 'confidence', 'competency_parce', 'prediction', 'label', 'path']
    df = df[[c for c in cols if c in df.columns]]

    csv_path = os.path.join(args.out_dir, "final_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()