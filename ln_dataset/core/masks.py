import torch
import torch.nn.functional as F
from ln_dataset.core.autoencoder import get_reconstruction_error

def _minmax01(x, eps=1e-6):
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    x = x / (x.amax(dim=(-2, -1), keepdim=True) + eps)
    return x

def _quantile_threshold(x, q):
    # q in [0,1], returns value t so that fraction q of values are <= t
    flat = x.reshape(-1)
    k = max(1, min(flat.numel(), int(q * flat.numel())))
    t, _ = torch.kthvalue(flat, k)
    return t

def _dilate(mask, k=7):
    # mask: (1,1,H,W) in {0,1}
    pad = k // 2
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)

def _gaussian_blur(mask, kernel_size=15, sigma=None):
    if sigma is None:
        sigma = kernel_size / 3.0
    x = torch.arange(kernel_size, device=mask.device).float() - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    k_h = gauss.view(1, 1, kernel_size, 1)
    k_w = gauss.view(1, 1, 1, kernel_size)
    pad = kernel_size // 2
    mask = F.conv2d(mask, k_h, padding=(pad, 0))
    mask = F.conv2d(mask, k_w, padding=(0, pad))
    return torch.clamp(mask, 0, 1)

def _ensemble_probs(models, x_norm):
    """
    Run ensemble forward passes *with* gradients enabled.
    """
    with torch.enable_grad():
        probs = []
        for m in models:
            logits = m(x_norm)
            probs.append(F.softmax(logits, dim=1))
        return torch.stack(probs, dim=0).mean(dim=0)  # (1,C)



def sensitivity_mask_from_models(img, models, mean, std):
    """
    Returns normalized gradient magnitude map (1,1,H,W) from ensemble confidence.
    Uses autograd.grad so we don't accumulate parameter grads.
    """
    img_req = img.detach().clone().requires_grad_(True)

    mean_t = torch.tensor(mean, device=img.device).view(1,3,1,1)
    std_t  = torch.tensor(std, device=img.device).view(1,3,1,1)
    x_norm = (img_req - mean_t) / std_t

    # forward with grad enabled
    probs = _ensemble_probs(models, x_norm)  # (1,1000)
    pred = probs.argmax(dim=1)              # (1,)
    score = torch.log(probs[0, pred] + 1e-12)

    grad = torch.autograd.grad(score, img_req, retain_graph=False, create_graph=False)[0]
    gmap = grad.abs().mean(dim=1, keepdim=True)  # (1,1,H,W)
    return _minmax01(gmap)

def generate_competency_mask_hybrid(
    ae_model,
    img_tensor,
    models,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    area=0.15,
    tau=0.25,
    alpha=1.0,
    beta=1.0,
    avoid_top_saliency=0.00,   # set e.g. 0.05 to avoid most-salient 5%
    contiguous=True,
    blur_k=15,
):
    """
    Hybrid competency mask: (low recon error) × (high confidence sensitivity).
    Produces a local contiguous patch by default.
    """
    # 1) Familiarity from AE reconstruction error
    err = get_reconstruction_error(ae_model, img_tensor)           # (1,1,H,W)
    err_n = _minmax01(err)
    fam = torch.exp(-err_n / max(tau, 1e-6))                       # high when error low

    # 2) Sensitivity from gradients of ensemble confidence
    sens = sensitivity_mask_from_models(img_tensor, models, mean, std)

    # Optional: avoid the extreme top-saliency pixels (protect semantic core)
    if avoid_top_saliency > 0:
        t = _quantile_threshold(sens, 1.0 - avoid_top_saliency)
        sens = sens * (sens < t).float()

    # 3) Combine
    score = (fam ** alpha) * (sens ** beta)
    score = _minmax01(score)

    # 4) Select region
    H, W = score.shape[-2:]
    target_pixels = int(area * H * W)

    if not contiguous:
        # top-area threshold
        t = _quantile_threshold(score, 1.0 - area)
        hard = (score >= t).float()
    else:
        # contiguous region-growing around best score
        allowed = (score >= _quantile_threshold(score, 0.85)).float()  # top 15% as corridor (adaptive-ish)
        # seed at argmax
        idx = score.view(-1).argmax()
        seed = torch.zeros_like(score)
        seed.view(-1)[idx] = 1.0
        hard = seed * allowed

        # grow until area reached (or stall)
        for _ in range(200):
            if hard.sum().item() >= target_pixels:
                break
            grown = _dilate(hard, k=9) * allowed
            if grown.sum().item() == hard.sum().item():
                # corridor too tight; relax once
                allowed = (score >= _quantile_threshold(score, 0.70)).float()
                grown = _dilate(hard, k=9) * allowed
                if grown.sum().item() == hard.sum().item():
                    break
            hard = grown

    # 5) Smooth edges (soft mask)
    soft = _gaussian_blur(hard, kernel_size=blur_k)
    return torch.clamp(soft, 0, 1)
