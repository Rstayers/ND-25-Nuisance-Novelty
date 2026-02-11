import argparse
import json
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os

from ln_dataset.core.configs import load_config
from ln_dataset.core.autoencoder import StandardAE
from ln_dataset.core.utils import ImgListDataset
from ln_dataset.core.generate_ln import ConfidenceJudge, select_mask

# Import nuisances
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance


def stable_seed(base, nuisance, p_idx, trial):
    import hashlib
    s = f"{base}_{nuisance}_{p_idx}_{trial}"
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 32)


def _apply_nuisance(nuisance_obj, img, mask, p, seed):
    try:
        return nuisance_obj.apply(img, mask, manual_param=p, seed=seed)
    except TypeError:
        return nuisance_obj.apply(img, mask, manual_param=p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--imglist', type=str, required=True)
    parser.add_argument('--ae_weights', type=str, required=True)
    parser.add_argument('--parce_calib', type=str, required=True)
    parser.add_argument('--save_json', type=str, default="bin_edges.json")

    # --- OPTIMIZED DEFAULTS ---
    parser.add_argument('--samples', type=int, default=500, help="Reduced to 500 for speed")
    parser.add_argument('--p_start', type=float, default=0.05)
    parser.add_argument('--p_end', type=float, default=1.0)
    parser.add_argument('--p_steps', type=int, default=10, help="Reduced to 10 steps")
    parser.add_argument('--trials_per_p', type=int, default=1, help="Reduced to 1 trial per step")

    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Settings: {args.samples} samples | {args.p_steps} steps | {args.trials_per_p} trials ---")

    # Init Models
    judge = ConfidenceJudge(device, args.ae_weights, args.parce_calib, cfg)
    ae = StandardAE().to(device)
    ae.load_state_dict(torch.load(args.ae_weights, map_location=device))
    ae.eval()

    # Init Data
    transform = transforms.Compose([
        transforms.Resize(tuple(cfg.image_size)),
        transforms.CenterCrop(tuple(cfg.image_size)),
        transforms.ToTensor(),
    ])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    # Subsample dataset
    if len(dataset) > args.samples:
        indices = np.random.choice(len(dataset), args.samples, replace=False)
    else:
        indices = np.arange(len(dataset))

    # Instantiate Nuisances
    nuisances = [
        LocalNoiseNuisance(),
        LocalPixelationNuisance(),
        LocalSpatialNuisance(),
        LocalPhotometricNuisance(mode='brightness'),
        LocalPhotometricNuisance(mode='contrast'),
        LocalPhotometricNuisance(mode='saturation'),
    ]

    valid_scores = []
    p_grid = np.linspace(args.p_start, args.p_end, args.p_steps)

    print(f"--- Calibrating Bins on {len(indices)} images ---")


    for idx in tqdm(indices):
        img, label, path = dataset[idx]
        if img is None: continue

        # Prepare Image
        img = img.unsqueeze(0).to(device)  # [1, C, H, W]
        base_name = os.path.splitext(os.path.basename(path))[0]


        with torch.no_grad():
            try:
                mask = select_mask(ae, img, label, target_area=0.33)
            except Exception:
                # If mask generation fails, skip this image
                continue

            if mask is None:
                continue
            for nuisance_obj in nuisances:
                consecutive_fails = 0
                has_success = False

                for p_i, p in enumerate(p_grid):
                    # Trial loop
                    for t in range(args.trials_per_p):
                        # Stable Seed
                        n_name = nuisance_obj.__class__.__name__
                        if hasattr(nuisance_obj, 'mode'): n_name += f"_{nuisance_obj.mode}"
                        seed = stable_seed(base_name, n_name, p_i, t)

                        try:
                            # Apply Nuisance
                            perturbed = _apply_nuisance(nuisance_obj, img, mask, p, seed)
                            if perturbed.ndim == 3: perturbed = perturbed.unsqueeze(0)

                            # Fast Check
                            score, pred, disagree, _ = judge.get_competency(perturbed, target_label=label)

                            if pred == label and not disagree:
                                valid_scores.append(float(score))
                                has_success = True
                                consecutive_fails = 0
                            else:
                                if has_success: consecutive_fails += 1

                        except Exception:
                            continue

                    # Strict Early Stopping
                    if has_success and consecutive_fails >= 5:
                        break

    # 3. Compute Quantiles
    print(f"Collected {len(valid_scores)} valid samples.")
    if len(valid_scores) == 0:
        print("Error: No valid samples found. Check mask selection or models.")
        return

    valid_scores = np.array(valid_scores)
    target_fracs = cfg.target_fracs
    cum_fracs = np.cumsum(target_fracs)

    edges = {}
    for i, frac in enumerate(cum_fracs[:-1]):
        cut = np.percentile(valid_scores, (1.0 - frac) * 100)
        edges[str(i + 1)] = float(cut)
    edges['5'] = 0.0

    print("Calculated Bin Edges:")
    print(json.dumps(edges, indent=4))

    with open(args.save_json, 'w') as f:
        json.dump(edges, f, indent=4)


if __name__ == "__main__":
    main()