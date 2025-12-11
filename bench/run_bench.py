import torch
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from bench.detectors import get_detector
from bench.loader import get_loaders
from bench.backbones import load_backbone


# --- Helper Metrics ---
def compute_id_threshold(confidences, tpr=0.95):
    """Calibrate threshold so 95% of ID data is accepted."""
    confidences = np.sort(confidences)
    cutoff_index = int(len(confidences) * (1 - tpr))
    return confidences[cutoff_index]


def get_outcome(is_correct, confidence, threshold):
    """Assigns the 4-way Taxonomy Label."""
    is_accepted = confidence >= threshold

    if is_correct and is_accepted:
        return "Clean_Success"
    if is_correct and not is_accepted:
        return "Nuisance_Novelty"  # The Goal
    if not is_correct and not is_accepted:
        return "Double_Failure"  # Safe Fail
    if not is_correct and is_accepted:
        return "Contained_Misidentification"  # Danger
    return "Error"


# --- Inference Engine ---
def run_inference(model, postprocessor, loader, device, is_id=False):
    """
    Runs inference and returns a list of dictionaries (one per image).
    """
    # Setup OpenOOD postprocessor (calculates statistics if needed)
    postprocessor.setup(model, loader, device)

    results = []

    print(f"Running Inference ({'ID' if is_id else 'LN'})...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue

            img = batch['image'].to(device)
            labels = batch['label'].to(device)

            # OpenOOD Inference: pred (index), conf (score)
            preds, confs = postprocessor.postprocess(model, img)

            # Move to CPU
            preds = preds.cpu().numpy()
            confs = confs.cpu().numpy()
            labels = labels.cpu().numpy()

            bs = len(preds)
            for i in range(bs):
                # Basic info
                row = {
                    'path': batch['path'][i],
                    'label': labels[i],
                    'prediction': preds[i],
                    'confidence': confs[i],
                    'correct_cls': int(preds[i] == labels[i])
                }

                # Add LN-specific metadata if not ID
                if not is_id:
                    row['level'] = batch['level'][i].item()
                    row['nuisance'] = batch['nuisance'][i]
                    row['competency_parce'] = batch['parce'][i].item()  # From generation
                else:
                    row['level'] = 0
                    row['nuisance'] = 'clean_id'
                    row['competency_parce'] = 1.0

                results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/largescale", help="Image Root")
    parser.add_argument('--manifest', default="", help="LN Manifest")
    parser.add_argument('--id_list', default="data/benchmark_imglist/imagenet/test_imagenet.txt", help="Clean ImageNet Val List")
    parser.add_argument('--backbones', nargs='+', default=['resnet50', 'vit_b_16'])
    parser.add_argument('--detectors', nargs='+', default=['msp', 'react', 'ash'])
    parser.add_argument('--out_dir', default="analysis/bench_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Data
    ln_loader, id_loader = get_loaders(args.data, args.manifest, args.id_list)

    all_rows = []

    # 2. Iterate Backbones
    for bb_name in args.backbones:
        print(f"\n=== Backbone: {bb_name} ===")
        model = load_backbone(bb_name, device)

        # 3. Iterate Detectors
        for det_name in args.detectors:
            print(f"--- Detector: {det_name} ---")
            detector = get_detector(det_name)

            # A. Process ID Data (To set Thresholds)
            id_results = run_inference(model, detector, id_loader, device, is_id=True)

            # Extract confidences to calculate threshold
            id_confs = [r['confidence'] for r in id_results]
            threshold = compute_id_threshold(id_confs, tpr=0.95)
            print(f" > Calibrated Threshold (95% TPR): {threshold:.4f}")

            # B. Process LN Data
            ln_results = run_inference(model, detector, ln_loader, device, is_id=False)

            # C. Merge & Classify Outcomes
            # We add backbone/detector info and the Outcome Type to every row
            for row in ln_results:
                row['backbone'] = bb_name
                row['detector'] = det_name
                row['threshold_used'] = threshold

                # THE CLASSIFICATION
                row['outcome'] = get_outcome(
                    is_correct=bool(row['correct_cls']),
                    confidence=row['confidence'],
                    threshold=threshold
                )

                all_rows.append(row)

    # 4. Save Huge CSV
    df = pd.DataFrame(all_rows)

    # Reorder columns for sanity
    cols = [
        'backbone', 'detector', 'nuisance', 'level',
        'outcome', 'correct_cls', 'confidence', 'competency_parce',
        'prediction', 'label', 'path', 'threshold_used'
    ]
    df = df[cols]

    csv_path = os.path.join(args.out_dir, "full_benchmark_dump.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nBenchmark Complete.")
    print(f"Saved {len(df)} rows to: {csv_path}")


if __name__ == "__main__":
    main()