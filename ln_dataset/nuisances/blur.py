import torch
import torchvision.transforms.functional as TF

class LocalBlurNuisance:
    def __init__(self, severity=1):
        self.severity = severity

    def apply(self, img_tensor, mask, manual_param=None):
        if manual_param is not None:
            # Scale severity to a visible sigma range (0.0 to 10.0)
            # Increased max sigma slightly to ensure 'Level 5' is very blurry
            sigma = manual_param * 10.0
        else:
            sigma = 0.0

        if sigma <= 0.1:
            return img_tensor

        # Calculate Kernel Size
        # Standard rule: k_size should be at least 6*sigma + 1 to capture the full Gaussian curve
        k_size = int(2 * int(3 * sigma) + 1)
        if k_size % 2 == 0: k_size += 1

        # Apply Gaussian Blur
        blurred = TF.gaussian_blur(img_tensor, kernel_size=k_size, sigma=[sigma, sigma])

        # Soft Blending is correct for Blur (fades from sharp to blurry)
        output = img_tensor * (1 - mask) + blurred * mask
        return output