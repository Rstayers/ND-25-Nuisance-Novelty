import math
from ln_dataset.core.utils import save_debug_maps
import argparse
import json
import os
import numpy as np
import torch
from torchvision import transforms, models
from tqdm import tqdm

# Import Core
from ln_dataset.core.autoencoder import StandardAE, get_reconstruction_error
from ln_dataset.core.masks import generate_reconstruction_mask
from ln_dataset.core.utils import ImgListDataset, save_tensor_as_img

# Import Nuisances
from ln_dataset.nuisances.noise import LocalNoiseNuisance
from ln_dataset.nuisances.pixel import LocalPixelationNuisance
from ln_dataset.nuisances.spatial import LocalSpatialNuisance
from ln_dataset.nuisances.photometric import LocalPhotometricNuisance

# Import Config
from ln_dataset.core.configs import load_config

# ==========================================
# 1. CONFIGURATION
# ==========================================

# --- CONSTANTS ---
DEBUG_MAX_IMAGES = 10
# Default Bins (will be overwritten if json provided)
BIN_EDGES = {'1': 1.0, '2': 0.75, '3': 0.5, '4': 0.25, '5': 0.0}
from collections import defaultdict
import hashlib, random

def stable_seed(*parts) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16) & 0x7FFFFFFF

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class GenerationStats:
    def __init__(self):
        self.total_source_images = 0
        self.successful_source_images = 0  # Images that produced at least one LN sample
        self.total_ln_generated = 0

        # Nested dict: stats[nuisance][level] = count
        self.distribution = defaultdict(lambda: defaultdict(int))

    def update(self, results):
        self.total_source_images += 1

        if not results:
            return

        self.successful_source_images += 1
        self.total_ln_generated += len(results)

        for res in results:
            n_name = res['nuisance']
            lvl = res['level']
            self.distribution[n_name][lvl] += 1

    def save(self, out_dir):
        # 1. Save Raw JSON
        data = {
            "summary": {
                "source_images_processed": self.total_source_images,
                "source_yield": self.successful_source_images,
                "yield_rate": self.successful_source_images / max(1, self.total_source_images),
                "total_ln_images_created": self.total_ln_generated
            },
            "breakdown": dict(self.distribution)
        }

        json_path = os.path.join(out_dir, "generation_stats.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\nStats saved to {json_path}")

        # 2. Print Pretty Table
        print("\n" + "=" * 40)
        print("GENERATION STATISTICS")
        print("=" * 40)
        print(f"Source Images: {self.total_source_images}")
        print(f"Yield (Valid): {self.successful_source_images} ({data['summary']['yield_rate']:.1%})")
        print(f"Total LN Files: {self.total_ln_generated}")
        print("-" * 40)
        print(f"{'Nuisance':<15} | {'L1':<5} {'L2':<5} {'L3':<5} {'L4':<5} {'L5':<5} | {'Total':<5}")
        print("-" * 40)

        for name in sorted(self.distribution.keys()):
            counts = self.distribution[name]
            row_sum = sum(counts.values())
            l1 = counts.get('1', 0)
            l2 = counts.get('2', 0)
            l3 = counts.get('3', 0)
            l4 = counts.get('4', 0)
            l5 = counts.get('5', 0)
            print(f"{name:<15} | {l1:<5} {l2:<5} {l3:<5} {l4:<5} {l5:<5} | {row_sum:<5}")
        print("=" * 40 + "\n")
def load_bin_edges_json(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)
    edges = payload.get("BIN_EDGES", payload)
    edges = {str(k): float(v) for k, v in edges.items()}
    for k in ["1", "2", "3", "4", "5"]:
        if k not in edges:
            raise ValueError(f"Missing BIN_EDGES['{k}'] in {path}")
    return edges

def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ==========================================
# 2. CONFIDENCE (PaRCE) JUDGE
# ==========================================



class ConfidenceJudge:
    def __init__(self, device, ae_weights_path, parce_calib_path, config):
        self.device = device
        self.config = config
        cal = torch.load(parce_calib_path, map_location=device)
        self.parce_method = cal.get("method", "overall")
        self.zscore = float(cal["zscore"])
        self.class_means = cal["class_means"].to(device).float()  # [C]
        self.class_stds = cal["class_stds"].to(device).float()  # [C]
        self.parce_eps = float(cal.get("eps", 1e-6))
        # 1. Load AE
        self.ae = StandardAE().to(device)
        state = torch.load(ae_weights_path, map_location=device)
        self.ae.load_state_dict(state)
        self.ae.eval()

        # 2. Load Competency Ensemble
        self.resnet = self._load_backbone("resnet50", config.models.resnet_ckpt)
        self.vit = self._load_backbone("vit_b_16", config.models.vit_ckpt)
        self.convnext = self._load_backbone("convnext_t", config.models.convnext_ckpt)
        self.densenet = self._load_backbone("densenet121", config.models.densenet_ckpt)

        # 3. Load PaRCE Stats (Legacy pipeline support)
        if os.path.exists(parce_calib_path):
            _ = torch.load(parce_calib_path, map_location=device)

    def _load_backbone(self, arch, ckpt_path):
        model = None
        if self.config.models.use_torchvision:
            if arch == "resnet50":
                model = models.resnet50(weights='IMAGENET1K_V1')
            elif arch == "vit_b_16":
                model = models.vit_b_16(weights='IMAGENET1K_V1')
            elif arch == "convnext_t":
                model = models.convnext_tiny(weights='IMAGENET1K_V1')
            elif arch == "densenet121":
                model = models.densenet121(weights='IMAGENET1K_V1')
        else:
            if arch == "resnet50":
                model = models.resnet50(num_classes=self.config.num_classes)
            elif arch == "vit_b_16":
                model = models.vit_b_16(num_classes=self.config.num_classes)
            elif arch == "convnext_t":
                model = models.convnext_tiny(num_classes=self.config.num_classes)
            elif arch == "densenet121":
                model = models.densenet121(num_classes=self.config.num_classes)

            if ckpt_path:
                state = torch.load(ckpt_path, map_location=self.device)
                if 'state_dict' in state: state = state['state_dict']
                model.load_state_dict(state)

        return model.to(self.device).eval()

    def get_competency(self, img, target_label):
        with torch.no_grad():
            mean = torch.tensor(self.config.mean, device=img.device).view(1, 3, 1, 1)
            std = torch.tensor(self.config.std, device=img.device).view(1, 3, 1, 1)
            img_norm = (img - mean) / std

            # classifiers get img_norm
            r_logits = self.resnet(img_norm)
            v_logits = self.vit(img_norm)
            c_logits = self.convnext(img_norm)
            d_logits = self.densenet(img_norm)
            ae_result =  self.ae(img)
            # autoencoder gets raw img in [0,1]
            recon = ae_result
            mse = torch.mean((recon - img) ** 2).item()

            # 2. Probs
            r_conf = torch.softmax(r_logits, dim=1)
            v_conf = torch.softmax(v_logits, dim=1)
            c_conf = torch.softmax(c_logits, dim=1)
            d_conf = torch.softmax(d_logits, dim=1)

            # 3. Preds
            r_pred = r_conf.argmax(dim=1).item()
            v_pred = v_conf.argmax(dim=1).item()
            c_pred = c_conf.argmax(dim=1).item()
            d_pred = d_conf.argmax(dim=1).item()

            # 5. AE Reconstruction
            # Paper Formula: C = Conf / (1 + lambda * MSE)
            # Conf is [0,1], Denom is >= 1.0, so C is always [0,1].


            # ensemble probabilities (p_hat_c)
            probs_ens = (r_conf + v_conf + c_conf + d_conf) / 4.0  # [1,C]
            pred = int(probs_ens.argmax(dim=1).item())
            p_hat = float(probs_ens.max(dim=1).values.item())

            # strict disagreement: every member must match GT label
            preds = [r_pred, v_pred, c_pred, d_pred]
            disagree = any(p != target_label for p in preds)

            # Eq.: p_id_c = 1 - Phi( (l - 2mu_c)/sigma_c - z )
            mu = self.class_means.view(1, -1)  # [1,C]
            sigma = self.class_stds.view(1, -1)  # [1,C]
            l = torch.tensor([[mse]], device=img.device).float()  # [1,1]
            z = torch.tensor(self.zscore, device=img.device).float()

            zvals = (l - 2.0 * mu) / (sigma + self.parce_eps) - z  # [1,C]
            p_id_c = 1.0 - normal_cdf(zvals)  # [1,C]

            # Eq. : rho = p_hat * Σ_c p_c * p_id_c
            sum_term = torch.sum(probs_ens * p_id_c, dim=1).item()
            parce = p_hat * sum_term

            return parce, pred, disagree, {"mse": mse, "p_hat": p_hat}

# ==========================================
# 3. GENERATION LOGIC
# ==========================================

def select_mask(ae_model, img, label, target_area=0.33):
    """
    Selects a mask using the simplified Grid Reconstruction method.

    Args:
        target_area (float): The percentage of the image to mask out (0.0 to 1.0).
                             Default is 0.33 (33%).
    """
    # 1. Generate the mask
    mask = generate_reconstruction_mask(ae_model, img, label, target_area=target_area)

    # 2. Check if mask is valid (not empty)
    if mask is None or mask.sum() == 0:
        return None

    return mask


def _assign_level(parce_score):
    """Maps standardized PaRCE score to Level 1-5 using BIN_EDGES."""
    if parce_score > BIN_EDGES['1']: return '1'
    if parce_score > BIN_EDGES['2']: return '2'
    if parce_score > BIN_EDGES['3']: return '3'
    if parce_score > BIN_EDGES['4']: return '4'

    # We do NOT reject samples that are harder than the last bin edge.
    return '5'

def _apply_nuisance(nuisance_obj, img, mask, p: float, seed: int):
    """
    Sticker should accept a seed; other nuisances will ignore it.
    Keep it backward-compatible.
    """
    try:
        return nuisance_obj.apply(img, mask, manual_param=p, seed=seed)
    except TypeError:
        return nuisance_obj.apply(img, mask, manual_param=p)


def sweep_and_select(judge, ae_model, img, label, output_dir, base_name, debug_dir=None):
    # 1) Mask
    mask = select_mask(ae_model, img, label)

    if debug_dir:
        # Generate reconstruction error map for visualization
        err = get_reconstruction_error(ae_model, img)
        err = (err - err.min()) / (err.max() - err.min() + 1e-6)
        save_debug_maps(img, err, mask, os.path.join(debug_dir, "viz_mask_selection"))

    # 2) Nuisances
    nuisances = [
        (LocalNoiseNuisance(), 'noise'),
        (LocalPixelationNuisance(), 'pixelation'),
        (LocalSpatialNuisance(), 'spatial'),
        (LocalPhotometricNuisance(mode='brightness'), 'brightness'),
        (LocalPhotometricNuisance(mode='contrast'), 'contrast'),
        (LocalPhotometricNuisance(mode='saturation'), 'saturation'),
    ]

    selected_samples = []

    # Sweep settings
    p_grid = np.linspace(0.05, 1.0, 50)

    # More trials for stochastic nuisances
    trials_by_name = {
        "noise": 2,
        "spatial": 2,
        "pixelation": 1,
        "brightness": 1,
        "contrast": 1,
        "saturation": 1,
    }

    # If we see sustained failures after having found some candidates, we can stop early
    max_consecutive_fail_after_success = 8

    for nuisance_obj, n_name in nuisances:
        best_candidates = {}  # lvl -> dict(img, score, p, seed)
        had_any_success = False
        consecutive_fail = 0
        n_trials = int(trials_by_name.get(n_name, 1))

        for pi, p in enumerate(p_grid):
            any_success_this_p = False

            for ti in range(n_trials):
                # Deterministic seed per (image, nuisance, p-index, trial)
                # Using base_name hash ensures consistency across runs for the same file
                seed = stable_seed(base_name, n_name, pi, ti)
                seed_all(seed)
                img_perturbed = _apply_nuisance(nuisance_obj, img, mask, float(p), seed)

                parce, pred, disagree, _ = judge.get_competency(img_perturbed, target_label=label)

                if disagree or (pred != label):
                    continue  # IMPORTANT: don't break; try another draw / next p

                any_success_this_p = True
                had_any_success = True
                consecutive_fail = 0

                lvl = _assign_level(float(parce))

                # If parsed level is None (too incompetent), skip
                if lvl is None:
                    continue

                # MINIMIZATION: keep the lowest PaRCE in this level (hardest sample in-bin)
                prev = best_candidates.get(lvl, None)
                if (prev is None) or (float(parce) < float(prev["score"])):
                    best_candidates[lvl] = {
                        "img": img_perturbed.detach().cpu(),  # Detach and move to CPU to save VRAM
                        "p": float(p),
                        "score": float(parce),
                        "seed": int(seed),
                    }

            if not any_success_this_p:
                if had_any_success:
                    consecutive_fail += 1
                    if consecutive_fail >= max_consecutive_fail_after_success:
                        break  # we’ve likely gone past the feasible region

        # Save + return paths
        for lvl, data in best_candidates.items():
            level_dir = os.path.join(output_dir, n_name, lvl)
            os.makedirs(level_dir, exist_ok=True)

            fname = f"{base_name}.png"
            save_path = os.path.join(level_dir, fname)

            # Using existing utility to save
            save_tensor_as_img(data["img"], save_path)

            rel_path = os.path.join(n_name, lvl, fname).replace("\\", "/")

            # Return rich metadata for stats
            selected_samples.append({
                "path": rel_path,
                "nuisance": n_name,
                "level": lvl,
                "p": data["p"],
                "parce": data["score"]
            })



    return selected_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--imglist', type=str, required=True)
    parser.add_argument('--ae_weights', type=str, required=True)
    parser.add_argument('--parce_calib', type=str, required=True)
    parser.add_argument('--bin_edges_json', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default="./ln_output")
    parser.add_argument('--debug_max', type=int, default=10)
    args = parser.parse_args()

    debug_maps = True
    cfg = load_config(args.config)
    print(f"Loaded Config: {cfg.name}")

    global BIN_EDGES
    if args.bin_edges_json and os.path.exists(args.bin_edges_json):
        BIN_EDGES = load_bin_edges_json(args.bin_edges_json)
        print(f"Loaded BIN_EDGES: {BIN_EDGES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init Judge
    judge = ConfidenceJudge(device, args.ae_weights, args.parce_calib, cfg)
    stats = GenerationStats()

    transform = transforms.Compose([
        transforms.Resize(tuple(cfg.image_size)),
        transforms.CenterCrop(tuple(cfg.image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    os.makedirs(args.out_dir, exist_ok=True)
    if debug_maps:
        os.makedirs(os.path.join(args.out_dir, "debug"), exist_ok=True)

    manifest_lines = []

    print(f"Starting Generation (Lite Mode - 0-1 Scores)...")

    for i in tqdm(range(len(dataset))):
        img, label, path = dataset[i]
        if img is None: continue

        img = img.unsqueeze(0).to(device)
        base_name = os.path.splitext(os.path.basename(path))[0]

        parce, pred, disagree, debug_info = judge.get_competency(img, target_label=label)

        if pred != label or disagree:
            stats.total_source_images += 1
            continue

        debug_dir = None
        if args.debug_max > 0 and i < args.debug_max:
            debug_dir = os.path.join(args.out_dir, "debug", f"{i:05d}_{base_name}")
            os.makedirs(debug_dir, exist_ok=True)
            save_tensor_as_img(img, os.path.join(debug_dir, "original.png"))

        results = sweep_and_select(judge, judge.ae, img, label, args.out_dir, base_name, debug_dir)
        stats.update(results)

        for res in results:
            manifest_lines.append(f"{res['path']} {label}")

    stats.save(args.out_dir)

    # --- Write the manifest file ---
    manifest_path = os.path.join(args.out_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines))
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()