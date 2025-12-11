import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class ClassifierAwareAE(nn.Module):
    def __init__(self):
        super(ClassifierAwareAE, self).__init__()

        # --- 1. Frozen Backbones (The "Classifier" Part) ---
        # We use both ResNet50 (Texture biased) and ViT (Shape biased)
        # to ensure the 'competency' is robust across architectures.

        # ResNet-50
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        # Extract 'layer3' -> Output is (B, 1024, 14, 14) for 224x224 input
        self.resnet_ext = create_feature_extractor(resnet, return_nodes={'layer3': 'feat'})

        # ViT-B/16
        vit = models.vit_b_16(weights='IMAGENET1K_V1')
        # Extract 'encoder.layers.encoder_layer_9' (Mid-to-Late Semantic features)
        # Output is (B, 197, 768) -> (CLS + 14x14 patches)
        self.vit_ext = create_feature_extractor(vit, return_nodes={'encoder.layers.encoder_layer_9': 'feat'})

        # Freeze Backbones
        self.resnet_ext.eval()
        self.vit_ext.eval()
        for p in self.resnet_ext.parameters(): p.requires_grad = False
        for p in self.vit_ext.parameters(): p.requires_grad = False

        # --- 2. Decoder (The "Reconstructor" Part) ---
        # Input Channels: 1024 (ResNet) + 768 (ViT) = 1792
        # Input Spatial: 14x14

        self.decoder = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(1792, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 28x28 -> 56x56
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 56x56 -> 112x112
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 112x112 -> 224x224
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Force output to [0, 1]
        )

        # Standard ImageNet Normalization Constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        x: [B, 3, 224, 224] in range [0, 1]
        """
        # 1. Normalize inputs for the backbones
        x_norm = (x - self.mean) / self.std

        # 2. Extract Features
        with torch.no_grad():
            # ResNet Features: [B, 1024, 14, 14]
            r_feat = self.resnet_ext(x_norm)['feat']

            # ViT Features: [B, 197, 768]
            v_out = self.vit_ext(x_norm)['feat']

            # Reshape ViT: Drop CLS token, reshape patches to spatial grid
            # [B, 197, 768] -> [B, 196, 768]
            v_patches = v_out[:, 1:, :]
            B, N, C = v_patches.shape
            H_grid = int(N ** 0.5)  # Should be 14

            # [B, 196, 768] -> [B, 768, 196] -> [B, 768, 14, 14]
            v_feat = v_patches.transpose(1, 2).reshape(B, C, H_grid, H_grid)

            # Concatenate Features
            combined_feat = torch.cat([r_feat, v_feat], dim=1)

        # 3. Decode
        recon = self.decoder(combined_feat)
        return recon


def get_reconstruction_error(ae_model, img_tensor):
    """
    Computes pixel-wise reconstruction error.
    Args:
        ae_model: ClassifierAwareAE model.
        img_tensor: Input image tensor [B, 3, H, W] in range [0, 1].
    """
    # Ensure input is safely [0, 1]
    img_input = torch.clamp(img_tensor, 0, 1)

    # Forward pass (AE handles normalization internally)
    with torch.no_grad():
        recon = ae_model(img_input)

    # Squared Error per pixel, averaged over channels
    # Output shape: (B, 1, H, W)
    error = (img_input - recon) ** 2
    error_map = error.mean(dim=1, keepdim=True)

    return error_map