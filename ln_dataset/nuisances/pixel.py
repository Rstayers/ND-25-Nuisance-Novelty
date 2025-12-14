import torch
import torch.nn.functional as F

class LocalPixelationNuisance:
    def __init__(self, severity=1):
        self.severity = severity
        # Ratios: 1.0 = Original, 0.05 = Max Blockiness
        self.ratios = [0.9, 0.7, 0.5, 0.3, 0.1]

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            manual_param: float [0.0, 1.0] representing SEVERITY.
                          0.0 = Ratio 1.0 (No change)
                          1.0 = Ratio ~0.02 (Huge blocks)
        """
        if manual_param is not None:
            # Map severity [0, 1] to Ratio [1.0, 0.02]
            ratio = 1.0 - (manual_param * 0.98)
            ratio = max(0.02, ratio)
        else:
            idx = max(0, min(self.severity - 1, 4))
            ratio = self.ratios[idx]

        # --- FIX: HARD MASK THRESHOLDING ---
        # We binarize the mask. Any region with >10% competency gets full pixelation.
        # This prevents the "ghosting" effect of alpha blending blocks over sharp details.
        mask_binary = (mask > 0.1).float()

        _, _, H, W = img_tensor.shape

        # Calculate downsampled dimensions
        new_h = max(1, int(H * ratio))
        new_w = max(1, int(W * ratio))

        # 1. Downsample using 'area' (Better for shrinking images than bilinear)
        small = F.interpolate(img_tensor, size=(new_h, new_w), mode='area')

        # 2. Upsample using 'nearest' to create the distinct pixel blocks
        pixelated = F.interpolate(small, size=(H, W), mode='nearest')

        # 3. Apply using the Binary Mask
        output = img_tensor * (1 - mask_binary) + pixelated * mask_binary
        return output