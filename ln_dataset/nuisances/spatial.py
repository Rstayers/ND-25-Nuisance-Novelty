import torch
import torch.nn.functional as F
import numpy as np
import cv2  # Ensure opencv-python-headless is installed


class LocalSpatialNuisance:
    def __init__(self, severity):
        self.severity = severity
        # (Alpha, Sigma) tuples
        # Alpha = Intensity of distortion
        # Sigma = Smoothness of distortion (Gaussian kernel size)
        self.params = [
            (20.0, 3.0),
            (40.0, 3.5),
            (60.0, 4.0),
            (80.0, 4.5),
            (100.0, 5.0)
        ]

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            manual_param: Can be a single float (alpha) or tuple (alpha, sigma).
                          If float, sigma defaults to 4.0.
        """
        if manual_param is not None:
            if isinstance(manual_param, (tuple, list)):
                alpha, sigma = manual_param
            else:
                alpha = manual_param
                sigma = 4.0  # Default smoothness
        else:
            idx = max(0, min(self.severity - 1, 4))
            alpha, sigma = self.params[idx]

        # Apply Elastic Transform
        output = self.elastic_transform(img_tensor, alpha, sigma)

        # Blend: Only warp the high-competency area
        output = img_tensor * (1 - mask) + output * mask
        return output

    def elastic_transform(self, image, alpha, sigma):
        _, _, H, W = image.shape

        # 1. Generate random displacement fields (Cpu numpy)
        # Seed logic can be added here for reproducibility if needed
        dx = np.random.rand(H, W) * 2 - 1
        dy = np.random.rand(H, W) * 2 - 1

        # 2. Smooth with Gaussian filter
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        # 3. Create meshgrid
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        # 4. Apply displacement
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        # 5. Remap image
        # Move tensor to CPU numpy for cv2.remap
        if image.is_cuda:
            img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.squeeze(0).permute(1, 2, 0).numpy()

        distorted_np = cv2.remap(
            img_np,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # 6. Back to tensor
        distorted = torch.from_numpy(distorted_np).permute(2, 0, 1).unsqueeze(0)
        return distorted.to(image.device)