import torch
import torchvision.transforms.functional as TF

class LocalBlurNuisance:
    def __init__(self, severity=1):
        self.severity = severity

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            manual_param: float [0.0, 1.0] representing SEVERITY.
                          Maps to Sigma [0.0, 8.0]
        """
        if manual_param is not None:
            # Scale severity to a visible sigma range (e.g., 0 to 8)
            sigma = manual_param * 8.0
        else:
            sigma = 0.0

        if sigma <= 0.0:
            return img_tensor

        # Kernel size must be odd.
        k_size = int(2 * int(2 * sigma) + 1)
        if k_size % 2 == 0: k_size += 1

        # Apply Gaussian Blur
        blurred = TF.gaussian_blur(img_tensor, kernel_size=k_size, sigma=[sigma, sigma])

        # Blend
        output = img_tensor * (1 - mask) + blurred * mask
        return output