import torch
import torch.nn as nn


class SimpleConvAE(nn.Module):
    def __init__(self):
        super(SimpleConvAE, self).__init__()
        # Encoder: Compresses 224x224 -> 50x50
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )
        # Decoder: Expands 50x50 -> 224x224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output forced to [0, 1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def get_reconstruction_error(ae_model, img_tensor):
    """
    Computes pixel-wise reconstruction error.
    Args:
        ae_model: Autoencoder model.
        img_tensor: Input image tensor [B, 3, H, W] in range [0, 1].
    """
    # FIX: Do not add mean/std. Input is already [0, 1].
    # Just clamp to be safe against minor floating point drift.
    img_input = torch.clamp(img_tensor, 0, 1)

    with torch.no_grad():
        recon = ae_model(img_input)

    # Squared Error per pixel, averaged over channels
    # Output shape: (B, 1, H, W)
    error = (img_input - recon) ** 2
    error_map = error.mean(dim=1, keepdim=True)

    return error_map