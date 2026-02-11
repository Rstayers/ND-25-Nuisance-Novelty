import torch
import numpy as np
from skimage.segmentation import felzenszwalb


def generate_reconstruction_mask(
        ae_model,
        img_tensor,
        scale=100,  # Felzenszwalb parameter (matches paper defaults)
        sigma=0.5,  # Felzenszwalb parameter
        min_size=50,  # Felzenszwalb parameter
        target_area=0.33
):
    """
    Generates a mask using the reconstruction loss method.

    Logic:
    1. Segment image into regions (Superpixels) using Felzenszwalb.
    2. For EACH segment:
       - Mask ONLY that segment (inpainting task).
       - Reconstruct the image.
       - Calculate MSE specifically on that segment's pixels.
    3. Select segments with the highest error.
    """
    ae_model.eval()
    B, C, H, W = img_tensor.shape
    assert B == 1, "Batch size must be 1"

    # 1. Get Segments
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    segments = felzenszwalb(img_np, scale=scale, sigma=sigma, min_size=min_size)

    unique_segments = np.unique(segments)
    segment_scores = []  #

    # 2. Iterate over segments
    for seg_id in unique_segments:
        # Create boolean mask for this segment
        seg_mask_np = (segments == seg_id)  # [H, W]

        # Create tensor mask (1.0 where the segment is, 0.0 elsewhere)
        # Inpainting usually implies filling with constant value or noise.
        mask_t = torch.from_numpy(seg_mask_np).float().to(img_tensor.device)
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Prepare Masked Input (Inpaint this segment)
        # We fill the segment with 0.5 (gray) or 0.0 (black) to hide it
        masked_input = img_tensor.clone()
        masked_input = masked_input * (1 - mask_t)  # Remove segment content

        # Reconstruct
        with torch.no_grad():
            recon = ae_model(masked_input)

        # 3. Compute Error ONLY on the masked segment
        # "compute the difference... this difference is the reconstruction loss"
        diff = (recon - img_tensor) ** 2

        # Mean over channels, then sum over the segment pixels
        diff = diff.mean(dim=1)  # [1, H, W]
        mse_segment = (diff * mask_t).sum() / mask_t.sum()  # Average error per pixel in segment

        segment_scores.append((seg_id, mse_segment.item(), mask_t.sum().item()))

    # 4. Selection Strategy
    # Sort segments by Error (Highest first)
    segment_scores.sort(key=lambda x: x[1], reverse=True)

    final_mask = torch.zeros((H, W), device=img_tensor.device)
    current_area = 0
    total_pixels = H * W
    target_pixels = total_pixels * target_area

    for seg_id, score, num_pix in segment_scores:
        if current_area >= target_pixels:
            break

        # Add this segment to the final mask
        seg_mask_np = (segments == seg_id)
        mask_t = torch.from_numpy(seg_mask_np).float().to(img_tensor.device)
        final_mask = torch.max(final_mask, mask_t)

        current_area += num_pix

    # Add channel dim for compatibility [1, 1, H, W]
    return final_mask.unsqueeze(0).unsqueeze(0)