import torch
import argparse
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Import your setup
from ln_dataset.generate_ln import ImgListDataset, ClassifierAwareAE, ConfidenceJudge
from ln_dataset.core.masks import generate_competency_mask_hybrid
from ln_dataset.nuisances import LocalSpatialNuisance, LocalPixelationNuisance, LocalNoiseNuisance
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance
from ln_dataset.utils import ImgListDataset


def calibrate_percentile_bins(judge, dataset, device, num_samples=1000, ae=None):
    print(f"\n--- Calibrating Bins (Percentile Method, n={num_samples}) ---")

    clean_scores = []  # Distribution of scores for clean images
    hard_scores = []  # Distribution of scores for the HARDEST valid image per sample

    # 1. Nuisances (Use a mix to probe model robustness)
    nuisances = [
        LocalNoiseNuisance(severity=1),
        LocalPixelationNuisance(severity=1), LocalSpatialNuisance(severity=1),
        LocalPhotometricNuisance(mode='brightness', severity=1),
        LocalPhotometricNuisance(mode='contrast', severity=1),
        LocalPhotometricNuisance(mode='saturation', severity=1)
    ]

    # 2. Select Random Subset
    total_imgs = len(dataset)
    indices = np.random.choice(total_imgs, min(num_samples, total_imgs), replace=False)

    # 3. Scan Loop
    for idx in tqdm(indices, desc="Scanning Distribution"):
        img, label, _ = dataset[idx]
        if img is None: continue
        img = img.unsqueeze(0).to(device)

        # --- A. Measure CLEAN Score ---
        score_clean, pred_clean, disagree, _ = judge.get_competency(img, target_label=label)

        if pred_clean == label and not disagree:
            clean_scores.append(score_clean)
        else:
            # If model is wrong on clean data, skip it
            continue

        # --- B. Find HARDEST Score (Mimic generate_ln loop) ---
        # We need to find the lowest score the model tolerates before breaking.
        mask = generate_competency_mask_hybrid(ae, img, models=[judge.resnet, judge.vit], area=0.20,
                                               avoid_top_saliency=0.15, contiguous=True)
        # We track the lowest score seen for THIS image across all valid nuisances
        current_img_lowest_valid_score = 1.0
        found_valid_perturbation = False

        for nuisance in nuisances:
            # Use same resolution as generator (important for consistency)
            # We scan forward; the moment it breaks, we stop this nuisance.
            # We want the score of the LAST working step.

            prev_score = score_clean

            for p in np.linspace(0.1, 1.0, 10):  # Coarser than generator for speed, but covers range
                modified_img = nuisance.apply(img, mask, manual_param=p)

                score, pred, disagree, _ = judge.get_competency(modified_img, target_label=label)

                # If broken, STOP. The *previous* p was the limit.
                if pred != label or disagree:
                    # The current image failed. The lowest valid score for this nuisance
                    # was the previous iteration (which we didn't store in this simple loop,
                    # but 'score' usually drops monotonically.
                    # Approximation: Use the current score but slightly penalized?
                    # Better: Just record valid scores.)
                    break

                # If valid, update the record
                if score < current_img_lowest_valid_score:
                    current_img_lowest_valid_score = score
                    found_valid_perturbation = True

        if found_valid_perturbation:
            hard_scores.append(current_img_lowest_valid_score)

    # 4. Compute Percentiles
    clean_scores = np.array(clean_scores)
    hard_scores = np.array(hard_scores)

    if len(hard_scores) == 0:
        print("Error: Model was robust to everything or failed everything. Check nuisances.")
        return

    # Strategy:
    # Level 1 (Easy): Median of Clean Data.
    # Level 5 (Hard): Bottom 20% of Hard Data.

    # Clean Anchor
    val_clean_median = np.percentile(clean_scores, 50)

    # Hard Anchors
    val_hard_p80 = np.percentile(hard_scores, 80)  # Top 20% of hard images (Level 2/3 boundary?)
    val_hard_p50 = np.percentile(hard_scores, 50)  # Median of hard images
    val_hard_p20 = np.percentile(hard_scores, 20)  # Bottom 20% of hard images (The Cliff edge)

    print("\n=== Calibration Stats ===")
    print(f"Clean Median:   {val_clean_median:.4f}")
    print(f"Hard P80:       {val_hard_p80:.4f}")
    print(f"Hard P50:       {val_hard_p50:.4f}")
    print(f"Hard P20:       {val_hard_p20:.4f} (Level 5 Ceiling)")

    # 5. Define Bins
    # We set the edges to ensure balanced population in the generated dataset.
    bins = {}

    # Edge 1: Above this is Level 1 (Clean-ish)
    bins['1'] = val_clean_median - 0.05  # Slightly relax clean median

    # Edge 4: Below this is Level 5 (The Danger Zone)
    # We use the 20th percentile of valid hard images.
    # This guarantees the bottom 20% of your generated data falls into Level 5.
    bins['4'] = val_hard_p20

    # Interpolate the middle
    # Level 2 (0.75 -> 0.55 approx)
    bins['2'] = bins['1'] - (bins['1'] - val_hard_p50) / 2

    # Level 3 (0.55 -> 0.20 approx)
    bins['3'] = val_hard_p50

    print("\n=== OPTIMAL BIN EDGES ===")
    print("Paste this into generate_ln.py:")
    print("-" * 30)
    print("BIN_EDGES = {")
    print(f"    '1': {bins['1']:.3f},  # > {bins['1']:.3f} = Level 1")
    print(f"    '2': {bins['2']:.3f},  # > {bins['2']:.3f} = Level 2")
    print(f"    '3': {bins['3']:.3f},  # > {bins['3']:.3f} = Level 3")
    print(f"    '4': {bins['4']:.3f},  # > {bins['4']:.3f} = Level 4")
    print(f"    '5': 0.000   # < {bins['4']:.3f} = Level 5")
    print("}")
    print("-" * 30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="data/images_largescale")
    parser.add_argument('--imglist', type=str, default="data/benchmark_imglist/imagenet/val_imagenet.txt")
    parser.add_argument('--ae_weights', type=str, default="ln_dataset/assets/ae_classifier_aware_weights.pth")
    parser.add_argument('--stats', type=str, default="ln_dataset/assets/parce_stats_full.pt")
    parser.add_argument('--samples', type=int, default=1000, help="Number of images to calibrate with")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = ClassifierAwareAE().to(device)
    ae.load_state_dict(torch.load(args.ae_weights, map_location=device))
    ae.eval()

    # Use the NEW ConfidenceJudge
    judge = ConfidenceJudge(device)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    calibrate_percentile_bins(judge, dataset, device, num_samples=args.samples, ae=ae)


if __name__ == "__main__":
    main()