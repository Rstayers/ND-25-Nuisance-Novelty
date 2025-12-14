import torch
import torch.nn.functional as F
import numpy as np
import cv2


class LocalSpatialNuisance:
    def __init__(self, severity=1):
        self.severity = severity
        # Tuples of (Alpha, Sigma)
        self.params = [
            (20.0, 3.0), (40.0, 3.5), (60.0, 4.0), (80.0, 4.5), (100.0, 5.0)
        ]

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            manual_param: float [0.0, 1.0] representing SEVERITY.
        """
        if manual_param is not None:
            # Alpha controls intensity of distortion
            alpha = manual_param * 150.0
            sigma = 4.0 + (manual_param * 2.0)
        else:
            idx = max(0, min(self.severity - 1, 4))
            alpha, sigma = self.params[idx]

        # Apply Elastic Transform with Flow Masking
        output = self.elastic_transform_masked(img_tensor, mask, alpha, sigma)
        return output

    def elastic_transform_masked(self, image, mask, alpha, sigma):
        _, _, H, W = image.shape

        # 1. Generate Random Flow Fields (dx, dy)
        dx = np.random.rand(H, W) * 2 - 1
        dy = np.random.rand(H, W) * 2 - 1

        # Smooth the noise to create elastic deformations
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        # 2. Prepare Mask for Flow Modulation
        # Convert mask to numpy [H, W] to match flow fields
        if mask.is_cuda:
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = mask.squeeze().numpy()

        # --- FIX: MASK THE FLOW, NOT THE PIXELS ---
        # We multiply the displacement (dx, dy) by the mask.
        # Where mask is 0, dx=0, so pixels stay put.
        # Where mask is 1, pixels move fully.
        dx = dx * mask_np
        dy = dy * mask_np

        # 3. Create Coordinate Grids
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        # Add displacement to coordinates
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        # 4. Remap (Warp)
        if image.is_cuda:
            img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.squeeze(0).permute(1, 2, 0).numpy()

        # Remap using the masked flow
        distorted_np = cv2.remap(
            img_np,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # Convert back to Torch
        distorted = torch.from_numpy(distorted_np).permute(2, 0, 1).unsqueeze(0).to(image.device)

        # No alpha blending needed here because the flow was already blended!
        return distorted