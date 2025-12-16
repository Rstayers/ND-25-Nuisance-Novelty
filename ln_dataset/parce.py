
import torch
import torch.nn.functional as F
import torch.distributions as dist
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import os
import numpy as np
from tqdm import tqdm

from ln_dataset.utils import STATS_FILE, NORM_MEAN, NORM_STD


class FullParceJudge:
    def __init__(self, device, ae_model, stats_path=STATS_FILE):
        self.device = device
        self.ae = ae_model

        # 1. Load Classifiers
        print("Loading classifiers...")
        self.resnet = models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()

        # Normalizer for classifiers
        self.normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

        # 2. Load Stats
        if os.path.exists(stats_path):
            data = torch.load(stats_path, map_location=device)
            self.mu_vec = data['mu'].to(device)  # [1000]
            self.sigma_vec = data['sigma'].to(device)  # [1000]
            self.z_calib = data['z_calib']  # Scalar
            self.accuracy = data['accuracy']
            print(f"Loaded Stats. Accuracy: {self.accuracy:.3f}, Global z: {self.z_calib:.3f}")
        else:
            print(f"WARNING: No stats at {stats_path}. Run --calibrate!")
            self.mu_vec = None
            self.z_calib = 0.0

    def get_competency(self, img_tensor, target_label=None):
        """
        Full PaRCE Estimation:
        Sum_c [ P(c|x) * P(ID | error, c) ]
        """
        with torch.no_grad():
            # A. CLASSIFIER PROBABILITIES (Requires Normalization)
            img_norm = self.normalize(img_tensor)

            logits_r = self.resnet(img_norm)
            logits_v = self.vit(img_norm)

            # Softmax to get P(c|x)
            probs_r = F.softmax(logits_r, dim=1)
            probs_v = F.softmax(logits_v, dim=1)

            # Ensemble Mixture
            probs_ens = (probs_r + probs_v) / 2.0  # Shape: [1, 1000]

            # Predictions for gating
            pred_idx = probs_ens.argmax(dim=1).item()
            disagreement = (probs_r.argmax(dim=1).item() != probs_v.argmax(dim=1).item())

            # B. RECONSTRUCTION ERROR (Raw Input [0,1])
            recon = self.ae(img_tensor)
            # MSE Scalar
            error = torch.mean((img_tensor - recon) ** 2).item()

            if self.mu_vec is None:
                return 0.5, pred_idx, disagreement, "Uncalibrated"

            # C. CALCULATE P(ID | c) FOR ALL CLASSES (Vectorized)
            # Formula: 1 - Phi( (error - 2*mu_c)/sigma_c - z )
            # We treat the "2*mu" as a shifting constant per user formula, or standard (err-mu).
            # We adhere to standard Z-score logic shifted by z_calib as per PaRCE general definition.
            # Z = (error - mu) / sigma

            # Broadcast error to [1000]
            err_vec = torch.tensor(error, device=self.device)

            # Calculate Z-scores for every class assumption
            # "How surprising is this error if the image actually belonged to class c?"
            z_scores = (err_vec - self.mu_vec) / (self.sigma_vec + 1e-9)

            # Apply Calibration shift
            # We want P(ID) to be high when error is low (Z is negative)
            # Survival Function: 1 - CDF(Z - z_calib)
            # If Z is large positive (high error), CDF -> 1, P(ID) -> 0.
            p_id_vec = 1.0 - dist.Normal(0, 1).cdf(z_scores - self.z_calib)

            # D. TOTAL PROBABILITY SUMMATION
            # P(Competency) = Sum_c ( P(c|x) * P(ID|c) )
            # probs_ens: [1, 1000], p_id_vec: [1000]
            parce_score = torch.sum(probs_ens[0] * p_id_vec).item()

            # Debug info for Target Class
            if target_label is not None:
                tgt = target_label
                debug_str = f"Err:{error:.4f} Mu:{self.mu_vec[tgt]:.4f} Z:{z_scores[tgt]:.2f} P(ID|y):{p_id_vec[tgt]:.3f}"
            else:
                debug_str = "N/A"

            return parce_score, pred_idx, disagreement, debug_str


# ==========================================
# 3. 2-STAGE CALIBRATION
# ==========================================
def calibrate_parce(ae_model, dataset, device, batch_size=32, max_samples=100000):
    print("--- Phase 1: Class-Conditional Statistics (Mu, Sigma) ---")

    # --- SUBSAMPLING LOGIC ---
    if len(dataset) > max_samples:
        print(f"Subsampling {max_samples} images from {len(dataset)} total (Random)...")
        # Use numpy to pick random indices
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        calib_dataset = Subset(dataset, indices)
    else:
        calib_dataset = dataset
    # -------------------------

    loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    ae_model.eval()

    class_errors = {}
    all_errors = []
    all_preds = []
    all_labels = []

    # Classifiers for Phase 2
    resnet = models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
    vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()
    normalizer = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader):
            imgs = imgs.to(device)

            # 1. AE Error
            recon = ae_model(imgs)
            mse = ((imgs - recon) ** 2).mean(dim=(1, 2, 3))

            # 2. Classifier Preds
            imgs_norm = normalizer(imgs)
            out_r = resnet(imgs_norm)
            out_v = vit(imgs_norm)
            probs = (F.softmax(out_r, dim=1) + F.softmax(out_v, dim=1)) / 2.0
            preds = probs.argmax(dim=1)

            for i in range(len(labels)):
                lbl = int(labels[i])
                err = mse[i].item()
                prd = int(preds[i])

                if lbl not in class_errors: class_errors[lbl] = []
                class_errors[lbl].append(err)

                all_errors.append(err)
                all_preds.append(prd)
                all_labels.append(lbl)

    # Compute Mu/Sigma per class
    mu_vec = torch.zeros(1000, device=device)
    sigma_vec = torch.ones(1000, device=device)

    print("Computing Class Statistics...")
    for lbl, errs in class_errors.items():
        arr = np.array(errs)
        mu_vec[lbl] = float(np.mean(arr))
        sigma_vec[lbl] = max(float(np.std(arr)), 1e-6)

    # --- Phase 2: Global Z Calibration ---
    print("\n--- Phase 2: Calibrating Global Z-Score ---")
    all_errors = torch.tensor(all_errors, device=device)
    all_preds = torch.tensor(all_preds, device=device)
    all_labels = torch.tensor(all_labels, device=device)

    correct = (all_preds == all_labels).float()
    accuracy = correct.mean().item()
    print(f"Model Accuracy on Calibration Set: {accuracy:.4f}")

    mus = mu_vec[all_labels]
    sigs = sigma_vec[all_labels]
    z_raw = (all_errors - mus) / sigs

    z_min, z_max = -10.0, 10.0
    best_z = 0.0
    min_diff = 1.0

    print("Optimizing z...")
    for z_curr in np.linspace(z_min, z_max, 1000):
        # 1 - CDF(z_raw - z_curr)
        p_ids = 1.0 - dist.Normal(0, 1).cdf(z_raw - z_curr)
        mean_competency = p_ids.mean().item()

        diff = abs(mean_competency - accuracy)
        if diff < min_diff:
            min_diff = diff
            best_z = z_curr

    print(f"Optimal Z found: {best_z:.3f} (Diff: {min_diff:.4f})")

    final_stats = {
        'mu': mu_vec,
        'sigma': sigma_vec,
        'z_calib': best_z,
        'accuracy': accuracy
    }

    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    torch.save(final_stats, STATS_FILE)
    print(f"Calibration Complete. Saved to {STATS_FILE}")