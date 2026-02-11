import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms, models

from ln_dataset.core.configs import load_config
from ln_dataset.core.utils import ImgListDataset
from ln_dataset.core.autoencoder import StandardAE


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    # standard normal CDF
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def load_backbone(arch: str, cfg, device: torch.device):
    arch = arch.lower()
    if cfg.models.use_torchvision:
        if arch == "resnet50":
            m = models.resnet50(weights="IMAGENET1K_V1")
        elif arch == "vit_b_16":
            m = models.vit_b_16(weights="IMAGENET1K_V1")
        elif arch == "convnext_t":
            m = models.convnext_tiny(weights="IMAGENET1K_V1")
        elif arch == "densenet121":
            m = models.densenet121(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unknown arch: {arch}")
        return m.to(device).eval()

    # custom checkpoints
    if arch == "resnet50":
        m = models.resnet50(num_classes=cfg.num_classes)
        ckpt = cfg.models.resnet_ckpt
    elif arch == "vit_b_16":
        m = models.vit_b_16(num_classes=cfg.num_classes)
        ckpt = cfg.models.vit_ckpt
    elif arch == "convnext_t":
        m = models.convnext_tiny(num_classes=cfg.num_classes)
        ckpt = cfg.models.convnext_ckpt
    elif arch == "densenet121":
        m = models.densenet121(num_classes=cfg.num_classes)
        ckpt = cfg.models.densenet_ckpt
    else:
        raise ValueError(f"Unknown arch: {arch}")

    if ckpt:
        state = torch.load(ckpt, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state)
    return m.to(device).eval()


@torch.no_grad()
def get_ensemble_probs_and_preds(models_list, img_norm):
    probs = []
    preds = []
    for m in models_list:
        p = torch.softmax(m(img_norm), dim=1)  # [1,C]
        probs.append(p)
        preds.append(int(p.argmax(dim=1).item()))
    probs_ens = torch.stack(probs, dim=0).mean(dim=0)  # [1,C]
    return probs_ens, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imglist", type=str, required=True)
    parser.add_argument("--ae_weights", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="parce_calib.pt")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=5.0)
    parser.add_argument("--z_step", type=float, default=0.05)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = cfg.num_classes
    eps = 1e-6

    print(f"Calibrating PaRCE (overall) for {cfg.name} with C={C}")

    # AE
    ae = StandardAE().to(device).eval()
    ae.load_state_dict(torch.load(args.ae_weights, map_location=device))

    # ensemble
    resnet = load_backbone("resnet50", cfg, device)
    vit = load_backbone("vit_b_16", cfg, device)
    convnext = load_backbone("convnext_t", cfg, device)
    densenet = load_backbone("densenet121", cfg, device)
    ens = [resnet, vit, convnext, densenet]

    # data (IMPORTANT: keep [0,1], no Normalize)
    transform = transforms.Compose([
        transforms.Resize(tuple(cfg.image_size)),
        transforms.CenterCrop(tuple(cfg.image_size)),
        transforms.ToTensor(),
    ])
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    if args.samples is not None and args.samples < len(dataset):
        idxs = np.random.choice(len(dataset), args.samples, replace=False)
    else:
        idxs = np.arange(len(dataset))

    mean = torch.tensor(cfg.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.std, device=device).view(1, 3, 1, 1)

    losses = []
    labels = []
    class_probs = []
    strict_correct = 0
    total = 0

    print(f"Collecting losses + probs on {len(idxs)} images...")
    for i in tqdm(idxs):
        img, y, _ = dataset[int(i)]
        if img is None:
            continue
        img = img.unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]
        y = int(y)

        img_norm = (img - mean) / std
        probs_ens, preds = get_ensemble_probs_and_preds(ens, img_norm)

        recon = ae(img)
        mse = torch.mean((recon - img) ** 2).item()

        # strict correctness: ALL members must equal GT label
        if all(p == y for p in preds):
            strict_correct += 1
        total += 1

        losses.append(mse)
        labels.append(y)
        class_probs.append(probs_ens.squeeze(0).detach().cpu().float().numpy())

    if total == 0:
        raise RuntimeError("No valid images loaded for calibration.")

    losses = np.asarray(losses, dtype=np.float32)        # [N]
    labels = np.asarray(labels, dtype=np.int64)          # [N]
    class_probs = np.asarray(class_probs, dtype=np.float32)  # [N,C]
    accuracy = strict_correct / total

    # Fit per-class Gaussian stats of reconstruction loss: mu_c, sigma_c
    mus = np.zeros(C, dtype=np.float32)
    sigmas = np.zeros(C, dtype=np.float32)

    global_mu = float(np.mean(losses))
    global_sigma = float(np.std(losses) + eps)

    for c in range(C):
        mask = (labels == c)
        if np.any(mask):
            mus[c] = float(np.mean(losses[mask]))
            sigmas[c] = float(np.std(losses[mask]) + eps)
        else:
            # if class absent in sample, fall back to global stats
            mus[c] = global_mu
            sigmas[c] = global_sigma

    # Select z to match Eq. (8): mean competency ≈ accuracy
    z_grid = np.arange(args.z_min, args.z_max + 1e-9, args.z_step, dtype=np.float32)

    # torch for stable CDF + broadcasting
    t_losses = torch.from_numpy(losses).float()[:, None]      # [N,1]
    t_probs = torch.from_numpy(class_probs).float()           # [N,C]
    t_mu = torch.from_numpy(mus).float()[None, :]             # [1,C]
    t_sigma = torch.from_numpy(sigmas).float()[None, :]       # [1,C]

    best_z = None
    best_diff = float("inf")
    best_mean = None

    for z in z_grid:
        zt = torch.tensor(float(z)).float()
        zvals = (t_losses - 2.0 * t_mu) / t_sigma - zt        # [N,C]
        p_id_c = 1.0 - normal_cdf(zvals)                      # Eq. (6)  [N,C]
        sum_term = torch.sum(p_id_c * t_probs, dim=1)         # Σ_c p_c * p_id_c
        p_hat = torch.max(t_probs, dim=1).values              # p_{ĉ}
        scores = p_hat * sum_term                             # Eq. (7)
        mean_score = float(torch.nanmean(scores).item())
        diff = abs(mean_score - accuracy)
        if diff < best_diff:
            best_diff = diff
            best_z = float(z)
            best_mean = mean_score

    print("\n=== PaRCE CALIBRATION (overall) ===")
    print(f"Strict accuracy:             {accuracy:.6f}")
    print(f"Chosen z-score:              {best_z:.2f}")
    print(f"Mean competency @ z:         {best_mean:.6f}")
    print(f"|mean competency - accuracy| {best_diff:.6f}")
    print("==================================\n")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(
        {
            "method": "overall",
            "num_classes": C,
            "zscore": best_z,
            "class_means": torch.from_numpy(mus),
            "class_stds": torch.from_numpy(sigmas),
            "eps": eps,
            "accuracy_strict": float(accuracy),
            "mean_competency": float(best_mean),
        },
        args.save_path,
    )
    print(f"Saved PaRCE calibration to: {args.save_path}")


if __name__ == "__main__":
    main()
