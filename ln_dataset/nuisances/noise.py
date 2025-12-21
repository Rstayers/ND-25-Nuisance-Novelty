import torch

class LocalNoiseNuisance:
    def __init__(self, severity=1):
        self.severity = severity
        self.epsilons = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]

    def apply(self, img_tensor, mask, gradient_tensor=None, manual_param=None):
        """
        Args:
            manual_param: float [0.0, 1.0] representing SEVERITY.
                          Maps to Epsilon [0.0, 0.3] (approx 75/255)
        """
        if manual_param is not None:
            # Map 0-1 to reasonable noise range 0.0 - 0.3
            eps = manual_param * 0.5
        else:
            idx = max(0, min(self.severity - 1, 4))
            eps = self.epsilons[idx]

        # Determine Noise Direction
        if gradient_tensor is None:
            noise = torch.randn_like(img_tensor)
        else:
            noise = -torch.sign(gradient_tensor)

        # Generate Perturbation
        perturbation = noise * eps

        # Apply
        output = img_tensor + perturbation * mask
        output = torch.clamp(output, 0, 1)
        return output