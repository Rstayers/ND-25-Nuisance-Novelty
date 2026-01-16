import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class ClassifierAwareAE(nn.Module):
    def __init__(self, resnet_path=None, vit_path=None):
        super(ClassifierAwareAE, self).__init__()

        # --- 1. Initialize Backbones (ImageNet Structure) ---
        print("Initializing Backbones...")
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        vit = models.vit_b_16(weights='IMAGENET1K_V1')

        # --- 2. Load Custom CUB Weights (If provided) ---
        if resnet_path:
            self._load_weights(resnet, resnet_path, "ResNet50")

        if vit_path:
            self._load_weights(vit, vit_path, "ViT-B/16")

        # --- 3. Extract Features ---
        # ResNet: Extract 'layer3' -> (B, 1024, 14, 14)
        self.resnet_ext = create_feature_extractor(resnet, return_nodes={'layer3': 'feat'})

        # ViT: Extract layer 9 -> (B, 197, 768)
        self.vit_ext = create_feature_extractor(vit, return_nodes={'encoder.layers.encoder_layer_9': 'feat'})

        # Freeze Backbones
        self.resnet_ext.eval()
        self.vit_ext.eval()
        for param in self.resnet_ext.parameters(): param.requires_grad = False
        for param in self.vit_ext.parameters(): param.requires_grad = False

        # --- 4. Decoder (Trainable) ---
        # Input: 1024 (ResNet) + 768 (ViT) = 1792 channels
        self.decoder = nn.Sequential(
            nn.Conv2d(1792, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 14 -> 28

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 28 -> 56

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 56 -> 112

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 112 -> 224

            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output 0-1
        )

    def _load_weights(self, model, path, name):
        print(f"Loading {name} weights from: {path}")
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location='cpu')

            # Handle if checkpoint is just state_dict or full dictionary
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Handle 'module.' prefix (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name_key = k[7:] if k.startswith('module.') else k
                new_state_dict[name_key] = v

            # Load with strict=False (ignore head/fc layer mismatches)
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"  > Loaded successfully. Missing keys (expected for Heads): {len(msg.missing_keys)}")

        except Exception as e:
            print(f"  > ERROR loading {name}: {e}")
            print("  > Continuing with ImageNet weights...")

    def forward(self, x):
        # 1. Normalize (assuming x is 0-1)
        # Standard ImageNet Mean/Std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x - mean) / std

        # 2. Extract Features
        with torch.no_grad():
            r_feat = self.resnet_ext(x_norm)['feat']
            v_out = self.vit_ext(x_norm)['feat']

            # Reshape ViT patches
            v_patches = v_out[:, 1:, :]  # Drop CLS
            B, N, C = v_patches.shape
            H_grid = int(N ** 0.5)
            v_feat = v_patches.transpose(1, 2).reshape(B, C, H_grid, H_grid)

            combined_feat = torch.cat([r_feat, v_feat], dim=1)

        # 3. Decode
        return self.decoder(combined_feat)


def get_reconstruction_error(ae_model, img_tensor):
    """Computes pixel-wise reconstruction error map."""
    recon = ae_model(img_tensor)
    # L2 distance squared per pixel
    diff = (recon - img_tensor) ** 2
    # Sum over channels -> (B, H, W)
    error_map = diff.sum(dim=1)
    return error_map