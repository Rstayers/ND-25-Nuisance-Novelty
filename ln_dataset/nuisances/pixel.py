import torch
import torch.nn.functional as F

class LocalPixelationNuisance:
    def __init__(self, severity=1):
        self.severity = severity
        # Downsample ratios (keep 90% ... keep 10%)
        self.ratios = [0.9, 0.7, 0.5, 0.3, 0.1]

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            manual_param: float [0.0, 1.0] representing SEVERITY.
                          0.0 = No change (Ratio 1.0)
                          1.0 = Max Pixelation (Ratio ~0.05)
        """
        if manual_param is not None:
            # Map severity [0, 1] to Ratio [1.0, 0.05]
            # We enforce a tiny minimum (0.02) to prevent 0x0 size errors
            ratio = 1.0 - (manual_param * 0.98)
            ratio = max(0.02, ratio)
        else:
            idx = max(0, min(self.severity - 1, 4))
            ratio = self.ratios[idx]

        _, _, H, W = img_tensor.shape
        new_h = max(1, int(H * ratio))
        new_w = max(1, int(W * ratio))

        # Downsample (Bilinear)
        small = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Upsample (Nearest Neighbor -> creates blocks)
        pixelated = F.interpolate(small, size=(H, W), mode='nearest')

        # Blend
        output = img_tensor * (1 - mask) + pixelated * mask
        return output