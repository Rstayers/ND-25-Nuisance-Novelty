import json
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import os
import argparse
import numpy as np
from tqdm import tqdm
import math

# Core Logic
from ln_dataset.core.autoencoder import ClassifierAwareAE
from ln_dataset.core.masks import generate_competency_mask_hybrid

# Nuisances
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance
from ln_dataset.utils import ImgListDataset, STATS_FILE, NORM_MEAN, NORM_STD, save_debug_maps, save_tensor_as_img, \
    tv_weights

# ==========================================
# 1. CONFIGURATION
# ==========================================

BIN_EDGES = {
    '1': 0.789,  # > 0.789 = Level 1
    '2': 0.756,  # > 0.756 = Level 2
    '3': 0.725,  # > 0.725 = Level 3
    '4': 0.671,  # > 0.671 = Level 4
    '5': 0.000   # <= 0.671 = Level 5
}
DEBUG_MAX_IMAGES = 30

def load_bin_edges_json(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)
    edges = payload.get("BIN_EDGES", payload)
    # normalize keys to strings "1".."5"
    edges = {str(k): float(v) for k, v in edges.items()}
    for k in ["1", "2", "3", "4"]:
        if k not in edges:
            raise ValueError(f"Missing BIN_EDGES['{k}'] in {path}")
    edges["5"] = 0.0
    return edges
def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def load_tv_model(name: str, device: torch.device):
    name = name.strip().lower()
    # Prefer enum weights if available; fallback to string for older torchvision.
    def _w(enum_path: str, fallback: str = "IMAGENET1K_V1"):
        try:
            mod, enum = enum_path.rsplit(".", 1)
            W = getattr(__import__(mod, fromlist=[enum]), enum)
            return getattr(W, "DEFAULT", None) or getattr(W, fallback)
        except Exception:
            return fallback

    if name in ["resnet50", "rn50"]:
        m = models.resnet50(weights=tv_weights(name))
    elif name in ["vit_b_16", "vitb16", "vit"]:
        m = models.vit_b_16(weights=tv_weights(name))
    elif name in ["convnext_t", "convnext_tiny", "cnext_tiny"]:
        m = models.convnext_tiny(weights=tv_weights(name))
    elif name in ["densenet121", "densenet"]:
        m = models.densenet121(weights=tv_weights(name))
    else:
        raise ValueError(f"Unknown model name: {name}")
    return m.to(device).eval()
# ==========================================
# 2. CONFIDENCE (PaRCE) JUDGE
# ==========================================
class ConfidenceJudge:
    def __init__(self, device,
                 ae_weights_path, parce_calib_path=STATS_FILE,
                 ensemble=("resnet50","vit_b_16", "convnext_t", "densenet121"), min_agree=1.0):
        self.device = device
        self.models = [load_tv_model(n, device) for n in ensemble]
        self.min_agree = float(min_agree)
        print("Loading classifiers and autoencoder...")
        self.resnet = models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()

        self.ae = ClassifierAwareAE().to(device)
        self.ae.load_state_dict(torch.load(ae_weights_path, map_location=device))
        self.ae.eval()

        self.normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        calib = torch.load(parce_calib_path, map_location="cpu")
        self.parce_means = calib["means"].to(device).float()
        self.parce_stds = calib["stds"].to(device).float()
        self.parce_z = float(calib["zscore"])
        self.parce_eps = 1e-6

    @torch.no_grad()
    def get_competency(self, img_tensor, target_label=None):
        img_norm = self.normalize(img_tensor)

        probs = []
        preds = []
        for m in self.models:
            p = torch.softmax(m(img_norm), dim=1)
            probs.append(p)
            preds.append(int(p.argmax(dim=1).item()))

        class_probs = torch.stack(probs, dim=0).mean(dim=0)  # [1,C]
        msp, pred_tensor = torch.max(class_probs, dim=1)
        pred_idx = int(pred_tensor.item())

        # majority vote agreement (vs strict unanimity)
        vote = max(set(preds), key=preds.count)
        agree_frac = preds.count(vote) / max(1, len(preds))
        disagree = agree_frac < self.min_agree

        rec_img = self.ae(img_tensor)
        if isinstance(rec_img, tuple):
            rec_img = rec_img[0]
        loss = torch.mean((rec_img - img_tensor) ** 2, dim=(1, 2, 3))

        means = self.parce_means[None, :]
        stds = (self.parce_stds[None, :] + self.parce_eps)
        zvals = (loss[:, None] - 2.0 * means) / stds - self.parce_z
        loss_probs = 1.0 - normal_cdf(zvals)

        scores = torch.sum(loss_probs * class_probs, dim=1)
        scores = scores * msp
        parce_score = float(scores.item())

        if target_label is not None:
            debug_str = f"PaRCE:{parce_score:.3f} MSP:{float(msp.item()):.3f} loss:{float(loss.item()):.4f} z:{self.parce_z:.2f}"
        else:
            debug_str = "N/A"

        return parce_score, pred_idx, disagree, debug_str


# ==========================================
# 3. GENERATION LOGIC
# ==========================================

def select_adaptive_mask(judge, ae_model, img, label):
    """
    Adaptive Mask Selection (v4):
    Iterates through progressively harder mask configurations.
    Returns the first mask that achieves 'reject-but-correct' behavior
    (low confidence, correct label) on a probe nuisance, or the best available.
    """

    # Schedule: Increase Area, Decrease Tau (sharpen familiarity), keep Saliency Avoidance low
    configs = [
        {'area': 0.10, 'tau': 0.20, 'avoid': 0.05, 'blur': 15, 'contig': True},
        {'area': 0.15, 'tau': 0.15, 'avoid': 0.05, 'blur': 15, 'contig': True},
        {'area': 0.20, 'tau': 0.10, 'avoid': 0.03, 'blur': 11, 'contig': True},
        {'area': 0.25, 'tau': 0.08, 'avoid': 0.00, 'blur': 9, 'contig': False},
        {'area': 0.30, 'tau': 0.05, 'avoid': 0.00, 'blur': 7, 'contig': False},
    ]

    best_mask = None
    best_score_impact = 0.0

    # Use Noise as the "Probe" nuisance to test mask efficacy
    probe_nuisance = LocalNoiseNuisance(severity=1)
    probe_p = 0.90  # stronger probe than 0.75
    require_edge = '4'

    for i, cfg in enumerate(configs):
        # 1. Generate Mask
        mask = generate_competency_mask_hybrid(
            ae_model,
            img,
            models=[judge.resnet, judge.vit],
            area=cfg['area'],
            tau=cfg['tau'],
            avoid_top_saliency=cfg['avoid'],
            contiguous=cfg['contig'],  # Enforce contiguous patches
            blur_k=cfg['blur']
        )

        # 2. Probe: Apply nuisance and check response
        img_probe = probe_nuisance.apply(img, mask, manual_param=probe_p)
        parce, pred, disagree, _ = judge.get_competency(img_probe, target_label=label)

        # 3. Evaluation
        is_correct = (pred == label) and (not disagree)

        # If we broke the model (flipped label), this mask might be too aggressive,
        # BUT if it's the first config, we have to keep it.
        # If we are correct and confidence is low (Level 4/5), this is perfect.

        if is_correct:
            if parce <= BIN_EDGES[require_edge]:  # Success: Reject-but-correct
                return mask  # Found an effective mask

            # If correct but confidence is still high, record this as a candidate
            # and try the next (harder) config.
            best_mask = mask
        else:
            # We flipped the label.
            # If this was a later config, the previous one was better.
            # If this is the first config, we'll store it but hope for better?
            # Actually, if we flip at 0.10, we likely can't do much.
            if best_mask is None:
                best_mask = mask  # Better than nothing

            # If we flipped, going harder (next config) is unlikely to help maintain correctness.
            # We stop here and use the previous best (or this one if it's all we have).
            break

    return best_mask


def sweep_and_select(judge, ae_model, img, label, output_dir, base_name, debug_dir=None):
    # 1. Adaptive Mask Generation (v4)
    # --------------------------------
    mask = select_adaptive_mask(judge, ae_model, img, label)

    if debug_dir is not None:
        save_debug_maps(judge, ae_model, img, label, mask, debug_dir)

    # 2. Define Nuisances
    # --------------------------------
    nuisances = [
        (LocalNoiseNuisance(severity=1), 'noise'),
        (LocalPixelationNuisance(severity=1), 'pixelation'),
        (LocalSpatialNuisance(severity=1), 'spatial'),
        (LocalPhotometricNuisance(mode='brightness', severity=1), 'brightness'),
        (LocalPhotometricNuisance(mode='contrast', severity=1), 'contrast'),
        (LocalPhotometricNuisance(mode='saturation', severity=1), 'saturation'),
    ]

    selected_samples = []

    for nuisance_obj, n_name in nuisances:
        best_candidates = {}

        # Sweep severity p (Forward: 0.05 -> 1.0)
        for p in np.linspace(0.05, 1.0, 50):

            # Apply Nuisance
            img_perturbed = nuisance_obj.apply(img, mask, manual_param=p)

            # Measure Competency
            parce, pred, disagree, _ = judge.get_competency(img_perturbed, target_label=label)

            # STOP if model fails (we want correct but low-confidence)
            if disagree or (pred != label):
                break

            # Identify Bin
            assigned_lvl = None
            if parce > BIN_EDGES['1']:
                assigned_lvl = '1'
            elif parce > BIN_EDGES['2']:
                assigned_lvl = '2'
            elif parce > BIN_EDGES['3']:
                assigned_lvl = '3'
            elif parce > BIN_EDGES['4']:
                assigned_lvl = '4'
            else:
                assigned_lvl = '5'

            # Update Candidate (Overwrite with harder 'p' for same level)
            if assigned_lvl:
                best_candidates[assigned_lvl] = {
                    'img': img_perturbed,
                    'p': p,
                    'score': parce,
                    'level': assigned_lvl
                }

        # Save Results
        for lvl, data in best_candidates.items():
            level_dir = os.path.join(output_dir, n_name, lvl)
            os.makedirs(level_dir, exist_ok=True)

            fname = f"{base_name}.png"
            save_path = os.path.join(level_dir, fname)

            save_tensor_as_img(data['img'], save_path)
            rel_path = os.path.join(n_name, lvl, fname).replace("\\", "/")
            selected_samples.append({'path': rel_path})

    return selected_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--imglist', type=str, required=True)
    parser.add_argument('--ae_weights', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="./ln_output_v4")
    parser.add_argument('--debug_maps', action='store_true')
    parser.add_argument('--debug_max', type=int, default=DEBUG_MAX_IMAGES)
    parser.add_argument('--bin_edges_json', type=str, default=None, help="Path to JSON produced by calibrate_bins.py (contains BIN_EDGES).")
    parser.add_argument('--parce_calib', type=str, default=STATS_FILE, help="Path to PaRCE calib .pt (means/stds/zscore).")

    args = parser.parse_args()
    global BIN_EDGES
    if args.bin_edges_json is not None:
        BIN_EDGES = load_bin_edges_json(args.bin_edges_json)
        print("Loaded BIN_EDGES from JSON:", BIN_EDGES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = ClassifierAwareAE().to(device)
    ae.load_state_dict(torch.load(args.ae_weights, map_location=device))
    ae.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    judge = ConfidenceJudge(device, ae_weights_path=args.ae_weights, parce_calib_path=args.parce_calib)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_lines = []

    print(f"Starting Generation v4 (Adaptive Masks)...")

    for i in tqdm(range(len(dataset))):
        img, label, path = dataset[i]
        if img is None: continue

        img = img.unsqueeze(0).to(device)
        base_name = os.path.splitext(os.path.basename(path))[0]

        parce, pred, disagree, debug_info = judge.get_competency(img, target_label=label)

        if i < DEBUG_MAX_IMAGES:
            print(f"\n[DEBUG Img {i}] L:{label} P:{pred} Dis:{disagree} PaRCE:{parce:.3f} | {debug_info}")

        if pred != label or disagree:
            continue

        debug_dir = None
        if args.debug_maps and i < args.debug_max:
            debug_dir = os.path.join(args.out_dir, "debug", f"{i:05d}_{base_name}")

        results = sweep_and_select(judge, ae, img, label, args.out_dir, base_name, debug_dir)
        for res in results:
            manifest_lines.append(f"{res['path']} {label}")

    with open(os.path.join(args.out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest_lines))

    print("\nGeneration Complete.")


if __name__ == "__main__":
    main()