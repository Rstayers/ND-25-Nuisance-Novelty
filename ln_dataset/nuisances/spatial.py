import torch
import torch.nn.functional as F
import numpy as np
import cv2


class LocalSpatialNuisance:
    def __init__(self, severity=1):
        self.severity = severity
        self.params = [
            (20.0, 3.0), (40.0, 3.5), (60.0, 4.0), (80.0, 4.5), (100.0, 5.0)
        ]

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            manual_param: float [0.0, 1.0] representing SEVERITY.
                          Maps to Alpha [0.0, 150.0]
        """
        if manual_param is not None:
            if isinstance(manual_param, (tuple, list)):
                alpha, sigma = manual_param
            else:
                # Map 0-1 to Alpha 0-150
                alpha = manual_param * 150.0
                sigma = 4.0 + (manual_param * 2.0)  # Scale sigma slightly too
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

        # CPU generation of random fields
        dx = np.random.rand(H, W) * 2 - 1
        dy = np.random.rand(H, W) * 2 - 1

        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(W), np.arange(H))

        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        if image.is_cuda:
            img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.squeeze(0).permute(1, 2, 0).numpy()

        distorted_np = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        distorted = torch.from_numpy(distorted_np).permute(2, 0, 1).unsqueeze(0).to(image.device)
        return distorted