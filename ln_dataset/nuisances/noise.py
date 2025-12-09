import torch


class LocalNoiseNuisance:
    def __init__(self, severity):
        self.severity = severity
        # Standard ImageNet robustness epsilons
        self.epsilons = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]

    def apply(self, img_tensor, mask, gradient_tensor=None, manual_param=None):
        """
        Args:
            img_tensor: (1, C, H, W)
            mask: (1, 1, H, W)
            gradient_tensor: (1, C, H, W) raw MSP gradients
            manual_param: float (epsilon), overrides severity.
        """
        if manual_param is not None:
            eps = manual_param
        else:
            idx = max(0, min(self.severity - 1, 4))
            eps = self.epsilons[idx]

        # Determine Noise Direction
        if gradient_tensor is None:
            # Fallback to random Gaussian if no gradients
            noise = torch.randn_like(img_tensor)
        else:
            # Competency Attack: Move AGAINST the gradient that maximizes MSP
            # (or use sign() for FGSM-style attack)
            noise = -torch.sign(gradient_tensor)

        # Generate Perturbation
        perturbation = noise * eps

        # Apply ONLY within the mask
        weighted_perturbation = perturbation * mask

        output = img_tensor + weighted_perturbation
        output = torch.clamp(output, 0, 1)

        return output