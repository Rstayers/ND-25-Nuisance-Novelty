import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torchvision import transforms, models
from tqdm import tqdm

from ln_dataset.core.autoencoder import ClassifierAwareAE
from ln_dataset.core.masks import generate_competency_mask_hybrid
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance
from ln_dataset.utils import ImgListDataset

# Import the SAME judge you use in generation
from ln_dataset.generate_ln import ConfidenceJudge

# ==========================================
# 1. BOOTSTRAP CONFIGURATION
# ==========================================
# These edges are used by the Adaptive Mask Selection logic to decide 
# when a mask successfully triggers "reject-but-correct" behavior.
# The calibration script will then refine these edges based on the data.
BIN_EDGES = {
    '1': 0.930,  # > 0.930 = Level 1
    '2': 0.908,  # > 0.908 = Level 2
    '3': 0.867,  # > 0.867 = Level 3
    '4': 0.749,  # > 0.749 = Level 4
    '5': 0.000  # <= 0.749 = Level 5
}


# -----------------------------
# Helpers
# -----------------------------
def assign_level(score: float, edges: Dict[str, float]) -> str:
    if score > edges["1"]:
        return "1"
    if score > edges["2"]:
        return "2"
    if score > edges["3"]:
        return "3"
    if score > edges["4"]:
        return "4"
    return "5"


def enforce_descending(e1: float, e2: float, e3: float, e4: float, eps: float = 1e-6) -> Tuple[
    float, float, float, float]:
    e1 = float(e1);
    e2 = float(e2);
    e3 = float(e3);
    e4 = float(e4)
    e2 = min(e2, e1 - eps)
    e3 = min(e3, e2 - eps)
    e4 = min(e4, e3 - eps)
    return e1, e2, e3, e4


def edges_from_quantiles(scores: np.ndarray, target_fracs: List[float]) -> Dict[str, float]:
    if len(target_fracs) != 5:
        raise ValueError("target_fracs must have 5 entries (for levels 1..5).")
    tf = np.array(target_fracs, dtype=np.float64)
    tf = tf / tf.sum()

    q1 = 1.0 - tf[0]
    q2 = 1.0 - (tf[0] + tf[1])
    q3 = 1.0 - (tf[0] + tf[1] + tf[2])
    q4 = 1.0 - (tf[0] + tf[1] + tf[2] + tf[3])

    q1 = float(np.clip(q1, 0.0, 1.0))
    q2 = float(np.clip(q2, 0.0, 1.0))
    q3 = float(np.clip(q3, 0.0, 1.0))
    q4 = float(np.clip(q4, 0.0, 1.0))

    e1, e2, e3, e4 = np.quantile(scores, [q1, q2, q3, q4])
    e1, e2, e3, e4 = enforce_descending(e1, e2, e3, e4)

    return {"1": float(e1), "2": float(e2), "3": float(e3), "4": float(e4), "5": 0.0}


@dataclass
class SweepRecord:
    nuisance: str
    ps: List[float]
    scores: List[float]
    last_valid_p: Optional[float]
    last_valid_score: Optional[float]


def simulate_saved_candidates(sweeps: List[SweepRecord], edges: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, int]]:
    selected_scores: List[float] = []
    counts = {str(i): 0 for i in range(1, 6)}

    for rec in sweeps:
        best: Dict[str, float] = {}
        for score in rec.scores:
            lvl = assign_level(score, edges)
            best[lvl] = score

        for lvl, sc in best.items():
            selected_scores.append(sc)
            counts[lvl] += 1

    if len(selected_scores) == 0:
        return np.array([], dtype=np.float32), counts

    return np.array(selected_scores, dtype=np.float32), counts


def pretty_edges(edges: Dict[str, float]) -> str:
    return (
        "BIN_EDGES = {\n"
        f"    '1': {edges['1']:.3f},  # > {edges['1']:.3f} = Level 1\n"
        f"    '2': {edges['2']:.3f},  # > {edges['2']:.3f} = Level 2\n"
        f"    '3': {edges['3']:.3f},  # > {edges['3']:.3f} = Level 3\n"
        f"    '4': {edges['4']:.3f},  # > {edges['4']:.3f} = Level 4\n"
        f"    '5': 0.000   # <= {edges['4']:.3f} = Level 5\n"
        "}\n"
    )


# -----------------------------
# Adaptive Mask Logic (Matches generate_ln.py v4)
# -----------------------------
def select_adaptive_mask(judge, ae_model, img, label):
    """
    Replicates the v4 generation logic:
    Try progressively harder masks until 'reject-but-correct' is achieved 
    (based on the bootstrap BIN_EDGES), or return the best attempt.
    """
    configs = [
        {'area': 0.10, 'tau': 0.20, 'avoid': 0.05, 'blur': 15},  # Medium
        {'area': 0.15, 'tau': 0.15, 'avoid': 0.05, 'blur': 15},  # Hard
        {'area': 0.20, 'tau': 0.10, 'avoid': 0.05, 'blur': 15},  # Aggressive
    ]

    best_mask = None

    # Use Noise as the "Probe" nuisance
    probe_nuisance = LocalNoiseNuisance(severity=1)
    probe_p = 0.75

    for i, cfg in enumerate(configs):
        mask = generate_competency_mask_hybrid(
            ae_model,
            img,
            models=[judge.resnet, judge.vit],
            area=cfg['area'],
            tau=cfg['tau'],
            avoid_top_saliency=cfg['avoid'],
            contiguous=True,
            blur_k=cfg['blur']
        )

        img_probe = probe_nuisance.apply(img, mask, manual_param=probe_p)
        parce, pred, disagree, _ = judge.get_competency(img_probe, target_label=label)

        is_correct = (pred == label) and (not disagree)

        if is_correct:
            # Check against bootstrap edges for "Level 4/5" success
            if parce <= BIN_EDGES['4']:
                return mask
            best_mask = mask
        else:
            if best_mask is None:
                best_mask = mask
            break

    return best_mask


# -----------------------------
# Main calibration routine
# -----------------------------
def collect_sweeps(
        judge,
        ae_model,
        dataset,
        device,
        indices: np.ndarray,
        p_start: float,
        p_end: float,
        p_steps: int,
) -> Tuple[List[float], List[SweepRecord]]:
    """
    Updated to use select_adaptive_mask.
    """
    nuisances = [
        (LocalNoiseNuisance(severity=1), "noise"),
        (LocalPixelationNuisance(severity=1), "pixelation"),
        (LocalSpatialNuisance(severity=1), "spatial"),
        (LocalPhotometricNuisance(mode="brightness", severity=1), "brightness"),
        (LocalPhotometricNuisance(mode="contrast", severity=1), "contrast"),
        (LocalPhotometricNuisance(mode="saturation", severity=1), "saturation"),
    ]

    p_grid = np.linspace(p_start, p_end, p_steps).astype(np.float32)
    clean_scores: List[float] = []
    sweeps: List[SweepRecord] = []

    for idx in tqdm(indices, desc="Collecting sweeps"):
        img, label, _ = dataset[int(idx)]
        if img is None:
            continue

        img = img.unsqueeze(0).to(device)

        # Clean gate
        score_clean, pred_clean, disagree_clean, _ = judge.get_competency(img, target_label=label)
        if pred_clean != label or disagree_clean:
            continue

        clean_scores.append(float(score_clean))

        # --- ADAPTIVE MASK SELECTION (v4) ---
        # This now matches the generator logic exactly.
        mask = select_adaptive_mask(judge, ae_model, img, label)

        for nuisance_obj, n_name in nuisances:
            valid_ps: List[float] = []
            valid_scores: List[float] = []

            last_valid_p: Optional[float] = None
            last_valid_score: Optional[float] = None

            for p in p_grid:
                img_pert = nuisance_obj.apply(img, mask, manual_param=float(p))
                score, pred, disagree, _ = judge.get_competency(img_pert, target_label=label)

                if disagree or (pred != label):
                    break

                valid_ps.append(float(p))
                valid_scores.append(float(score))
                last_valid_p = float(p)
                last_valid_score = float(score)

            if len(valid_scores) > 0:
                sweeps.append(
                    SweepRecord(
                        nuisance=n_name,
                        ps=valid_ps,
                        scores=valid_scores,
                        last_valid_p=last_valid_p,
                        last_valid_score=last_valid_score,
                    )
                )

    return clean_scores, sweeps


def calibrate_bins_precise(
        judge,
        ae_model,
        dataset,
        device,
        samples: int,
        seed: int,
        p_start: float,
        p_end: float,
        p_steps: int,
        target_fracs: List[float],
        iters: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    total_imgs = len(dataset)
    n = min(samples, total_imgs)
    indices = rng.choice(total_imgs, n, replace=False)

    clean_scores, sweeps = collect_sweeps(
        judge=judge,
        ae_model=ae_model,
        dataset=dataset,
        device=device,
        indices=indices,
        p_start=p_start,
        p_end=p_end,
        p_steps=p_steps,
    )

    if len(sweeps) == 0:
        raise RuntimeError("No valid sweeps collected. Check that clean images are correct + judge agrees.")

    all_valid_scores = np.concatenate([np.array(rec.scores, dtype=np.float32) for rec in sweeps], axis=0)
    edges = edges_from_quantiles(all_valid_scores, target_fracs)

    for t in range(iters):
        selected_scores, counts = simulate_saved_candidates(sweeps, edges)
        if selected_scores.size == 0:
            raise RuntimeError("Selection simulation produced zero candidates; edges are too strict?")

        new_edges = edges_from_quantiles(selected_scores, target_fracs)

        total_sel = int(selected_scores.size)
        props = {k: (counts[k] / max(1, total_sel)) for k in counts}
        print(f"\n[Iter {t + 1}/{iters}] Selected candidates: {total_sel}")
        print("  Level proportions:", {k: f"{props[k]:.3f}" for k in ["1", "2", "3", "4", "5"]})
        print("  Edges:", {k: f"{new_edges[k]:.4f}" for k in ["1", "2", "3", "4"]})

        delta = max(abs(new_edges[k] - edges[k]) for k in ["1", "2", "3", "4"])
        edges = new_edges
        if delta < 1e-4:
            print(f"  Converged (max edge change {delta:.2e}).")
            break

    if len(clean_scores) > 0:
        cs = np.array(clean_scores, dtype=np.float32)
        print("\n=== Clean score sanity ===")
        print(
            f"Clean median: {np.median(cs):.4f} | p10: {np.quantile(cs, 0.10):.4f} | p90: {np.quantile(cs, 0.90):.4f}")
        print(f"Final edge1:  {edges['1']:.4f} (Level-1 threshold)")

    return edges


def construct_judge(device, ae_weights: str, parce_calib: Optional[str] = None):
    kw = {"ae_weights_path": ae_weights}
    if parce_calib is not None:
        for k in ["parce_calib_path", "parce_calib", "stats_path", "stats"]:
            try:
                return ConfidenceJudge(device, **kw, **{k: parce_calib})
            except TypeError:
                pass
    return ConfidenceJudge(device, **kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/images_largescale")
    parser.add_argument("--imglist", type=str, default="data/benchmark_imglist/imagenet/val_imagenet.txt")
    parser.add_argument("--ae_weights", type=str, default="ln_dataset/assets/ae_classifier_aware_weights.pth")
    parser.add_argument("--parce_calib", type=str, default=None)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--p_start", type=float, default=0.05)
    parser.add_argument("--p_end", type=float, default=1.0)
    parser.add_argument("--p_steps", type=int, default=50)

    # Note: Mask params (area, avoid, contiguous) are now handled by the adaptive schedule.

    parser.add_argument(
        "--target_fracs",
        type=str,
        default="0.2,0.2,0.2,0.2,0.2",
        help="Comma-separated fractions for levels 1..5 (will be normalized).",
    )
    parser.add_argument("--iters", type=int, default=5, help="Iterations for selection-simulation refinement")
    parser.add_argument("--save_json", type=str, default=None)

    args = parser.parse_args()

    target_fracs = [float(x) for x in args.target_fracs.split(",")]
    if len(target_fracs) != 5:
        raise ValueError("--target_fracs must have exactly 5 comma-separated values (levels 1..5).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = ClassifierAwareAE().to(device)
    ae.load_state_dict(torch.load(args.ae_weights, map_location=device))
    ae.eval()

    judge = construct_judge(device, args.ae_weights, parce_calib=args.parce_calib)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    print("\n--- Calibrating BIN_EDGES (Adaptive v4) ---")
    edges = calibrate_bins_precise(
        judge=judge,
        ae_model=ae,
        dataset=dataset,
        device=device,
        samples=args.samples,
        seed=args.seed,
        p_start=args.p_start,
        p_end=args.p_end,
        p_steps=args.p_steps,
        target_fracs=target_fracs,
        iters=args.iters,
    )

    print("\n=== OPTIMAL BIN EDGES ===")
    print("Paste this into generate_ln.py:")
    print("-" * 30)
    print(pretty_edges(edges).rstrip())
    print("-" * 30)

    if args.save_json is not None:
        payload = {"BIN_EDGES": edges, "meta": vars(args)}
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved edges JSON to: {args.save_json}")


if __name__ == "__main__":
    main()