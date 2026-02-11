import torch.nn as nn


class StandardAE(nn.Module):
    """
    A standard Image-to-Image Autoencoder
    Trained to reconstruct the input image from itself (denoising/inpainting).
    """

    def __init__(self, in_channels=3, latent_dim=512):
        super(StandardAE, self).__init__()

        # Encoder: Compresses image to latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 112
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 14
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # Decoder: Reconstructs image from latent representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 28
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 56
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 112
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),  # 224
            nn.Sigmoid()  # Output pixels in [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def get_reconstruction_error(ae_model, img_tensor):
    """Computes pixel-wise reconstruction error map."""
    recon = ae_model(img_tensor)
    # L2 distance squared per pixel
    diff = (recon - img_tensor) ** 2
    # Sum over channels -> (B, H, W)
    error_map = diff.sum(dim=1)
    return error_map