import torch
import torchvision.transforms.functional as TF


class LocalBlurNuisance:
    def __init__(self, severity):
        self.severity = severity

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            manual_param: float (sigma), range [0.0, 5.0]
        """
        if manual_param is not None:
            sigma = manual_param
        else:
            sigma = 0.0

        if sigma <= 0.0:
            return img_tensor

        # Kernel size must be odd.
        # Rule of thumb: k = 2 * ceil(2*sigma) + 1
        k_size = int(2 * int(2 * sigma) + 1)
        if k_size % 2 == 0: k_size += 1

        # Apply Gaussian Blur
        blurred = TF.gaussian_blur(img_tensor, kernel_size=k_size, sigma=[sigma, sigma])

        # Blend
        output = img_tensor * (1 - mask) + blurred * mask
        return output