# ln_dataset/core/masks.py
import torch
import torch.nn.functional as F


def generate_competency_mask(saliency_map, severity):
    """
    Converts a raw saliency map into a soft blending mask based on severity.

    Severity Controls:
    1. Percentile (How much of the object do we attack?)
    2. Softness (How smooth are the edges?)

    Args:
        saliency_map: (H, W) tensor in [0, 1].
        severity: Int [1-5].

    Returns:
        mask: (1, 1, H, W) tensor in [0, 1] ready for blending.
    """
    # Severity Configuration
    # (Percentile Threshold, Gaussian Blur Kernel Size)
    # Sev 1: Attack top 10% (The eyes/most critical part)
    # Sev 5: Attack top 60% (The whole body)
    configs = {
        1: (0.90, 3),
        2: (0.80, 5),
        3: (0.70, 7),
        4: (0.60, 9),
        5: (0.40, 11)
    }

    threshold_q, blur_k = configs[severity]

    # 1. Thresholding: Find the value at the q-th percentile
    # Flatten and sort to find pixel value threshold
    flat = saliency_map.view(-1)
    k = int(threshold_q * flat.numel())
    if k >= flat.numel(): k = flat.numel() - 1
    val, _ = torch.kthvalue(flat, k)

    # 2. Binarize (Softly)
    # Create binary mask where saliency > threshold
    mask = (saliency_map > val).float()

    # 3. Gaussian Blur for Soft Edges (Crucial for ECCV "Naturalism")
    # Add dimensions for conv2d: (N, C, H, W)
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Create simple gaussian kernel
    kernel_size = blur_k
    sigma = kernel_size / 3.0
    x = torch.arange(kernel_size).float() - kernel_size // 2
    x = x.to(mask.device)
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    # Separable convolution (H then W)
    k_h = gauss_1d.view(1, 1, kernel_size, 1)
    k_w = gauss_1d.view(1, 1, 1, kernel_size)

    padding = kernel_size // 2
    mask = F.conv2d(mask, k_h, padding=(padding, 0))
    mask = F.conv2d(mask, k_w, padding=(0, padding))

    # Clamp to ensure [0,1]
    mask = torch.clamp(mask, 0.0, 1.0)

    return mask