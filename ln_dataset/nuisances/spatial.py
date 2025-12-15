import torch
import torch.nn.functional as F
import numpy as np
import cv2


class LocalSpatialNuisance:
    def __init__(self, severity=1):
        self.severity = severity
        # ImageNet-C parameters for Elastic Transform
        # Format: (alpha_multiplier, sigma_multiplier, affine_multiplier)
        # Note: These are significantly "rougher" than standard data aug settings.
        self.params = [
            (0, 0, 0),  # 0: No-op
            (0.5, 0.2, 0.1),  # 1
            (1.5, 0.3, 0.2),  # 2
            (2.5, 0.4, 0.3),  # 3
            (3.5, 0.5, 0.4),  # 4
            (4.5, 0.6, 0.5)  # 5
        ]

    def apply(self, img_tensor, mask, manual_param=None):
        """
        Args:
            img_tensor: Tensor [B, C, H, W] (values 0-1)
            mask: Tensor [B, 1, H, W] (values 0 or 1)
            manual_param: float [0.0, 1.0] representing SEVERITY.
        """
        # Determine parameters
        if manual_param is not None:
            # Linearly interpolate between severity 1 and 5 params based on manual_param
            s = manual_param
            alpha_c = 0.5 + (s * 4.0)  # 0.5 -> 4.5
            sigma_c = 0.2 + (s * 0.4)  # 0.2 -> 0.6
            affine_c = 0.1 + (s * 0.4)  # 0.1 -> 0.5
        else:
            idx = max(0, min(self.severity, 5))
            alpha_c, sigma_c, affine_c = self.params[idx]

        # Scale constants relative to image height (robust to resolution)
        _, _, H, W = img_tensor.shape
        alpha = alpha_c * H
        sigma = sigma_c * H * 0.05
        alpha_affine = affine_c * H * 0.05

        return self.elastic_transform_imagenet_c(img_tensor, mask, alpha, sigma, alpha_affine)

    def elastic_transform_imagenet_c(self, image, mask, alpha, sigma, alpha_affine):
        """
        Replicates ImageNet-C elastic transform logic with masking.
        """
        B, C, H, W = image.shape
        device = image.device

        # ---------------------------------------------------------
        # 1. Random Affine (The ImageNet-C addition)
        # ---------------------------------------------------------
        center_x, center_y = W // 2, H // 2

        # Source points
        pts1 = np.float32([[center_x - H // 4, center_y - H // 4],
                           [center_x + H // 4, center_y - H // 4],
                           [center_x - H // 4, center_y + H // 4]])

        # Perturb points to get Destination points
        perturbation = np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        pts2 = pts1 + perturbation

        # Get Affine Matrix (2x3)
        M = cv2.getAffineTransform(pts1, pts2)
        M_torch = torch.from_numpy(M).unsqueeze(0).to(device).float()  # [1, 2, 3]

        # ---------------------------------------------------------
        # 2. Generate Random Elastic Flow Fields (dx, dy)
        # ---------------------------------------------------------
        noise_x = torch.rand(1, 1, H, W, device=device) * 2 - 1
        noise_y = torch.rand(1, 1, H, W, device=device) * 2 - 1

        # Gaussian blur the noise to get smooth flow
        k_size = int(sigma * 4) | 1
        dx = self._gaussian_blur(noise_x, k_size, sigma) * alpha
        dy = self._gaussian_blur(noise_y, k_size, sigma) * alpha

        # ---------------------------------------------------------
        # 3. Apply Masking to Combine Affine + Elastic
        # ---------------------------------------------------------
        mask_float = mask.float()

        # Create identity grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')

        # Flatten grid for Matrix Multiplication with Affine Matrix
        ones = torch.ones_like(grid_x).flatten()
        # FIX 1: Ensure coords are float for matmul
        coords = torch.stack([grid_x.flatten(), grid_y.flatten(), ones], dim=1).T.float()  # [3, H*W]

        # Apply Affine Transform to grid coordinates
        affine_coords = torch.matmul(M_torch, coords)  # [1, 2, H*W]
        affine_x = affine_coords[0, 0].view(H, W)
        affine_y = affine_coords[0, 1].view(H, W)

        # Calculate how much the affine transform moved things
        diff_affine_x = affine_x - grid_x
        diff_affine_y = affine_y - grid_y

        # Combine displacements (Affine Shift + Elastic Wiggle)
        # Note: .squeeze() is safe here assuming B=1 for flow generation or shared flow
        total_dx = diff_affine_x + dx.squeeze()
        total_dy = diff_affine_y + dy.squeeze()

        # MASKING: Only apply displacement where mask is active (1.0)
        final_dx = total_dx * mask_float.squeeze()
        final_dy = total_dy * mask_float.squeeze()

        # ---------------------------------------------------------
        # 4. Create Final Sampling Grid
        # ---------------------------------------------------------
        # Normalize coordinates to [-1, 1] for grid_sample
        new_grid_x = 2.0 * (grid_x + final_dx) / (W - 1) - 1.0
        new_grid_y = 2.0 * (grid_y + final_dy) / (H - 1) - 1.0

        # FIX 2: Use dim=-1 to stack (works for 2D [H,W] or 3D [B,H,W])
        grid = torch.stack((new_grid_x, new_grid_y), dim=-1)

        # Ensure grid is [B, H, W, 2]
        if grid.ndim == 3:  # Currently [H, W, 2]
            grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        elif grid.ndim == 4 and grid.shape[0] != B:
            grid = grid.expand(B, -1, -1, -1)

        # ---------------------------------------------------------
        # 5. Sample
        # ---------------------------------------------------------
        # padding_mode='reflection' is key for ImageNet-C style destruction
        distorted = F.grid_sample(image, grid, mode='bilinear', padding_mode='reflection', align_corners=True)

        return distorted

    def _gaussian_blur(self, x, k_size, sigma):
        # Simple torch gaussian blur wrapper
        if k_size % 2 == 0: k_size += 1
        pad = k_size // 2
        # Create 1D kernel
        range_x = torch.arange(k_size, device=x.device).float() - pad
        kernel_1d = torch.exp(-0.5 * (range_x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_1d = kernel_1d.view(1, 1, -1, 1)  # [OC, IC, H, W]

        # Separable convolution
        x = F.conv2d(x, kernel_1d.transpose(2, 3), padding=(0, pad), groups=1)
        x = F.conv2d(x, kernel_1d, padding=(pad, 0), groups=1)
        return x