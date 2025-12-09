import torch
import torchvision.transforms.functional as TF


class LocalPhotometricNuisance:
    def __init__(self, mode, severity):
        """
        Args:
            mode: 'brightness', 'contrast', 'saturation'
            severity: 1-5 (Used only if manual_param is None)
        """
        self.mode = mode
        self.severity = severity

        # Default Lookups (Pre-calibration placeholders)
        self.params = {
            'brightness': [0.15, 0.25, 0.35, 0.45, 0.55],  # Additive shift
            'contrast': [0.85, 0.70, 0.55, 0.40, 0.25],  # Factor (1.0 = Original)
            'saturation': [0.80, 0.60, 0.40, 0.20, 0.00],  # Factor (0.0 = Grayscale)
        }

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            img_tensor: (1, C, H, W) normalized or [0,1]
            mask: (1, 1, H, W) competency mask
            manual_param: float, overrides the severity lookup.
        """
        # 1. Determine the Strength Factor
        if manual_param is not None:
            factor = manual_param
        else:
            # Fallback to severity lookup
            # Clamp severity 1-5 to index 0-4
            idx = max(0, min(self.severity - 1, 4))
            factor = self.params[self.mode][idx]

        # 2. Generate Nuisance Image
        if self.mode == 'brightness':
            # Additive brightness (glare)
            nuisance_img = torch.clamp(img_tensor + factor, 0, 1)

        elif self.mode == 'contrast':
            # Move towards mean (grey)
            mean = img_tensor.mean()
            nuisance_img = (img_tensor - mean) * factor + mean
            nuisance_img = torch.clamp(nuisance_img, 0, 1)

        elif self.mode == 'saturation':
            # TF.adjust_saturation expects tensor in [0,1]
            nuisance_img = TF.adjust_saturation(img_tensor, factor)

        # 3. Blend based on Competency Mask
        output = img_tensor * (1 - mask) + nuisance_img * mask
        return output