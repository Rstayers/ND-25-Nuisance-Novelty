import torch
import torch.nn.functional as F
from ln_dataset.core.autoencoder import get_reconstruction_error


def generate_competency_mask(ae_model, img_tensor, percentile=0.5):
    """
    Generates a mask targeting High Competency regions (Low Reconstruction Error).

    Args:
        ae_model: The autoencoder.
        img_tensor: The input image tensor.
        percentile: The fraction of the image to target (0.5 = top 50% most familiar).

    Returns:
        mask: (1, 1, H, W) Soft mask, 1.0 = High Competency (Target for Nuisance).
    """
    # 1. Get Reconstruction Error
    # High Error = Novel/Complex (Don't touch)
    # Low Error = Familiar/Simple (Touch these to tank confidence)
    error_map = get_reconstruction_error(ae_model, img_tensor)

    # 2. Invert Error to get Competency
    # We want mask to be HIGH where error is LOW.
    # Normalize error to [0, 1]
    flat_err = error_map.view(-1)

    # Find cutoff for "Low Error"
    # We want the bottom X percentile of error.
    k = int(percentile * flat_err.numel())
    threshold_val, _ = torch.kthvalue(flat_err, k)

    # Create Binary Mask: 1 where Error <= Threshold
    mask = (error_map <= threshold_val).float()

    # 3. Smooth the mask (Soft edges prevent artifacts)
    # Gaussian Blur
    k_size = 15
    sigma = k_size / 3.0
    mask = gaussian_blur(mask, k_size, sigma)

    return mask


def gaussian_blur(mask, kernel_size, sigma):
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size).float() - kernel_size // 2
    x = x.to(mask.device)
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # Reshape for separable conv
    k_h = gauss.view(1, 1, kernel_size, 1)
    k_w = gauss.view(1, 1, 1, kernel_size)

    padding = kernel_size // 2
    mask = F.conv2d(mask, k_h, padding=(padding, 0))
    mask = F.conv2d(mask, k_w, padding=(0, padding))
    return torch.clamp(mask, 0, 1)