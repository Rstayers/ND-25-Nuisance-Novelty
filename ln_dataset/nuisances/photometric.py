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
            # Fallback (Should be unused in optimization loop)
            factor = 1.0

        # 2. Generate Nuisance Image
        if self.mode == 'brightness':
            # Additive brightness (0.0 to 1.0)
            # Literature Fig 17: "Increased Brightness"
            nuisance_img = torch.clamp(img_tensor + factor, 0, 1)

        elif self.mode == 'contrast':
            # Multiplicative (1.0 -> 0.0)
            # Literature Fig 16: "Reduced Contrast"
            mean = img_tensor.mean()
            nuisance_img = (img_tensor - mean) * factor + mean
            nuisance_img = torch.clamp(nuisance_img, 0, 1)

        elif self.mode == 'saturation':
            # Saturation Factor (1.0 -> 3.0+)
            # Literature Fig 20: "Increased Saturation"
            # TF.adjust_saturation expects shape [..., H, W]
            nuisance_img = TF.adjust_saturation(img_tensor, factor)
            nuisance_img = torch.clamp(nuisance_img, 0, 1)

        # 3. Blend: Only apply to High Competency regions
        output = img_tensor * (1 - mask) + nuisance_img * mask
        return output