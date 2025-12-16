import torch
import torchvision.transforms.functional as TF


class LocalPhotometricNuisance:
    def __init__(self, mode='contrast', severity=1):
        self.mode = mode
        self.severity = severity

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            manual_param: float [0.0, 1.0] SEVERITY.
        """
        # 1. Determine the Strength Factor
        if manual_param is not None:
            p = manual_param
        else:
            p = 0.0  # No op fallback

        # 2. Generate Nuisance Image
        if self.mode == 'brightness':
            # Severity 1.0 = Add 0.6 brightness (Very washout)
            factor = p * 3.0
            nuisance_img = torch.clamp(img_tensor + factor, 0, 1)

        elif self.mode == 'contrast':
            # Severity 1.0 = Factor 0.1 (Almost Grey)
            # Severity 0.0 = Factor 1.0 (Original)
            factor = 1.0 + (p * 10.0)
            mean = img_tensor.mean()
            nuisance_img = (img_tensor - mean) * factor + mean
            nuisance_img = torch.clamp(nuisance_img, 0, 1)

        elif self.mode == 'saturation':
            # Severity 1.0 = Factor 3.0 (Oversaturated)
            # Severity 0.0 = Factor 1.0
            factor = 1.0 + (p * 10.0)
            nuisance_img = TF.adjust_saturation(img_tensor, factor)

        else:
            return img_tensor

        # Blend
        output = img_tensor * (1 - mask) + nuisance_img * mask
        return output