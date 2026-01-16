import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from ln_dataset.configs import load_config
from ln_dataset.core.autoencoder import ClassifierAwareAE
from ln_dataset.core.masks import generate_competency_mask_hybrid
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance
from ln_dataset.utils import ImgListDataset
from ln_dataset.generate_ln import ConfidenceJudge

# ==========================================
# 1. BOOTSTRAP CONFIGURATION
# ==========================================
# Initial guesses for bin edges used ONLY during the mask selection phase.
# The final edges will be overwritten by this script's output.
BIN_EDGES = {
    '1': 0.930,
    '2': 0.908,
    '3': 0.867,
    '4': 0.749,
    '5': 0.000
}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def enforce_descending(e1, e2, e3, e4, eps=1e-6):
    e1, e2, e3, e4 = float(e1), float(e2), float(e3), float(e4)
    e2 = min(e2, e1 - eps)
    e3 = min(e3, e2 - eps)
    e4 = min(e4, e3 - eps)
    return e1, e2, e3, e4


def edges_from_quantiles(scores: np.ndarray, target_fracs: List[float]) -> Dict[str, float]:
    """Calculates edges such that Level 5 floor is ALWAYS 0.0"""
    if len(target_fracs) != 5:
        raise ValueError("target_fracs must have 5 entries.")

    tf = np.array(target_fracs, dtype=np.float64)
    tf = tf / tf.sum()

    # Calculate quantiles for the top 4 bins
    q1 = 1.0 - tf[0]
    q2 = 1.0 - (tf[0] + tf[1])
    q3 = 1.0 - (tf[0] + tf[1] + tf[2])
    q4 = 1.0 - (tf[0] + tf[1] + tf[2] + tf[3])

    qs = [float(np.clip(x, 0.0, 1.0)) for x in [q1, q2, q3, q4]]

    e1, e2, e3, e4 = np.quantile(scores, qs)
    e1, e2, e3, e4 = enforce_descending(e1, e2, e3, e4)

    # FIX: Force Level 5 floor to 0.0 to capture all hard samples
    return {"1": float(e1), "2": float(e2), "3": float(e3), "4": float(e4), "5": 0.0}


def assign_level(score: float, edges: Dict[str, float]) -> str:
    if score > edges["1"]: return "1"
    if score > edges["2"]: return "2"
    if score > edges["3"]: return "3"
    if score > edges["4"]: return "4"
    return "5"  # All remaining valid scores go to 5


@dataclass
class SweepRecord:
    nuisance: str
    ps: List[float]
    scores: List[float]


def select_adaptive_mask(judge, ae_model, img, label):
    """
    Selects a mask that creates a 'hard' but 'correct' image if possible.
    Uses judge.resnet and judge.vit as the visual saliency ensemble.
    """
    configs = [
        {'area': 0.10, 'tau': 0.20, 'avoid': 0.05, 'blur': 15, 'contig': True},
        {'area': 0.15, 'tau': 0.15, 'avoid': 0.05, 'blur': 15, 'contig': True},
        {'area': 0.20, 'tau': 0.10, 'avoid': 0.03, 'blur': 11, 'contig': True},
        {'area': 0.25, 'tau': 0.08, 'avoid': 0.00, 'blur': 9, 'contig': False},
        {'area': 0.30, 'tau': 0.05, 'avoid': 0.00, 'blur': 7, 'contig': False},
    ]

    best_mask = None
    probe_nuisance = LocalNoiseNuisance(severity=1)
    probe_p = 0.75

    for cfg in configs:
        mask = generate_competency_mask_hybrid(
            ae_model, img,
            models=[judge.resnet, judge.vit],  # Use judge members explicitly
            area=cfg['area'], tau=cfg['tau'],
            avoid_top_saliency=cfg['avoid'], contiguous=cfg['contig'], blur_k=cfg['blur']
        )

        img_probe = probe_nuisance.apply(img, mask, manual_param=probe_p)
        parce, pred, disagree, _ = judge.get_competency(img_probe, target_label=label)

        if (pred == label) and (not disagree):
            # If we achieved Level 4/5 difficulty with correctness, stop.
            if parce <= BIN_EDGES['4']:
                return mask
            best_mask = mask
        else:
            if best_mask is None: best_mask = mask
            break

    return best_mask


# ==========================================
# 3. CORE LOGIC
# ==========================================

def collect_sweeps(judge, ae_model, dataset, device, indices, p_grid):
    nuisances = [
        (LocalNoiseNuisance(severity=1), "noise"),
        (LocalPixelationNuisance(severity=1), "pixelation"),
        (LocalSpatialNuisance(severity=1), "spatial"),
        (LocalPhotometricNuisance(mode="brightness", severity=1), "brightness"),
        (LocalPhotometricNuisance(mode="contrast", severity=1), "contrast"),
        (LocalPhotometricNuisance(mode="saturation", severity=1), "saturation"),

    ]

    sweeps = []

    print(f"Sweeping {len(indices)} images to find realistic score distributions...")
    for idx in tqdm(indices):
        img, label, _ = dataset[int(idx)]
        if img is None: continue
        img = img.unsqueeze(0).to(device)

        # Skip already broken images
        _, pred_clean, disagree_clean, _ = judge.get_competency(img, target_label=label)
        if pred_clean != label or disagree_clean: continue

        # Get Mask
        mask = select_adaptive_mask(judge, ae_model, img, label)

        # Sweep Nuisances
        for nuisance_obj, n_name in nuisances:
            valid_ps, valid_scores = [], []

            for p in p_grid:
                img_pert = nuisance_obj.apply(img, mask, manual_param=float(p))
                score, pred, disagree, _ = judge.get_competency(img_pert, target_label=label)

                # Collect only CORRECT samples (this is the key: Level 5 must be correct)
                if not disagree and pred == label:
                    valid_ps.append(float(p))
                    valid_scores.append(float(score))
                else:
                    # Once we break the image, harder p won't help
                    break

            if valid_scores:
                sweeps.append(SweepRecord(n_name, valid_ps, valid_scores))

    return sweeps


def calibrate_bins_precise(judge, ae_model, dataset, device, args):
    # Setup
    p_grid = np.linspace(args.p_start, args.p_end, args.p_steps)
    target_fracs = [float(x) for x in args.target_fracs.split(",")]

    # Sample Indices
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), min(args.samples, len(dataset)), replace=False)

    # 1. Collect Valid Perturbed Scores
    sweeps = collect_sweeps(judge, ae_model, dataset, device, indices, p_grid)
    if not sweeps:
        raise RuntimeError("No valid perturbed samples found. Check data/model.")

    all_scores = np.concatenate([np.array(r.scores) for r in sweeps])

    # 2. Initial Edges
    edges = edges_from_quantiles(all_scores, target_fracs)

    # 3. Refine (Simulation)
    # Simulate the "best sample selection" logic to ensure bins are balanced
    # when only ONE sample per image/nuisance is kept.
    for t in range(args.iters):
        selected = []
        counts = {str(k): 0 for k in range(1, 6)}

        for rec in sweeps:
            # Simulate selection: Find the hardest sample (lowest score) in the hardest bin available
            best_in_sweep = None
            best_lvl_int = 0

            # Map all scores in this sweep to current levels
            for s in rec.scores:
                lvl = assign_level(s, edges)
                lvl_int = int(lvl)

                # We prefer higher Level number (Harder) -> then Lowest Score
                if lvl_int > best_lvl_int:
                    best_lvl_int = lvl_int
                    best_in_sweep = s
                elif lvl_int == best_lvl_int:
                    if best_in_sweep is None or s < best_in_sweep:
                        best_in_sweep = s

            if best_in_sweep is not None:
                selected.append(best_in_sweep)
                counts[str(best_lvl_int)] += 1

        selected = np.array(selected)
        new_edges = edges_from_quantiles(selected, target_fracs)

        # Check convergence
        diff = sum(abs(new_edges[k] - edges[k]) for k in ['1', '2', '3', '4'])
        edges = new_edges
        print(f"[Iter {t + 1}] Counts: {dict(sorted(counts.items()))} | Edge4: {edges['4']:.3f}")
        if diff < 1e-4: break

    return edges


# ==========================================
# 4. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    # Standard Pipeline Arguments
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--parce_calib', type=str, required=True)
    parser.add_argument('--ae_weights', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--imglist', type=str, required=True)
    parser.add_argument('--save_json', type=str, default="bin_edges.json")

    # Tuning Arguments (defaults matched to your preference)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--p_start', type=float, default=0.05)
    parser.add_argument('--p_end', type=float, default=1.0)
    parser.add_argument('--p_steps', type=int, default=30)
    parser.add_argument('--target_fracs', type=str, default="0.2,0.2,0.2,0.2,0.2")
    parser.add_argument('--iters', type=int, default=5)

    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init Models
    judge = ConfidenceJudge(device, args.ae_weights, args.parce_calib, cfg)
    ae = ClassifierAwareAE().to(device)
    ae.load_state_dict(torch.load(args.ae_weights, map_location=device))
    ae.eval()

    # Init Data (No Normalize for AE access, Judge normalizes internally)
    transform = transforms.Compose([
        transforms.Resize(tuple(cfg.image_size)),
        transforms.CenterCrop(tuple(cfg.image_size)),
        transforms.ToTensor(),
    ])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    print("\n--- Running Precise Bin Calibration (Sweeping Nuisances) ---")
    edges = calibrate_bins_precise(judge, ae, dataset, device, args)

    print("\n=== FINAL BIN EDGES ===")
    print(json.dumps(edges, indent=4))

    with open(args.save_json, 'w') as f:
        json.dump(edges, f, indent=4)
    print(f"Saved to {args.save_json}")


if __name__ == "__main__":
    main()