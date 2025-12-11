import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Import specific model classes for isinstance checks
from torchvision.models.resnet import ResNet
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models.swin_transformer import SwinTransformer
from torchvision.models.convnext import ConvNeXt
from torchvision.models.densenet import DenseNet


class OpenOODWrapper(nn.Module):
    """
    Universal wrapper to make Torchvision models compatible with OpenOOD Postprocessors.
    Handles differences in feature extraction (avgpool vs cls token) and head names.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

        # --- 1. SPECIAL CASE: ConvNeXt ---
        # ConvNeXt.classifier = Sequential(LayerNorm2d, Flatten, Linear)
        # We must split this because LayerNorm expects 4D, but OpenOOD needs 2D features.
        if isinstance(model, ConvNeXt):
            # We treat the Norm and Flatten as part of "feature extraction"
            self.fc = model.classifier[2]  # The Linear Layer
            self.convnext_norm = model.classifier[0]  # LayerNorm2d
            self.convnext_flatten = model.classifier[1]  # Flatten
            self.is_convnext = True
        else:
            self.is_convnext = False
            # Standard Auto-detection
            if hasattr(model, 'fc'):
                self.fc = model.fc
            elif hasattr(model, 'heads'):
                self.fc = model.heads
            elif hasattr(model, 'head'):
                self.fc = model.head
            elif hasattr(model, 'classifier'):
                self.fc = model.classifier
            else:
                raise AttributeError("Could not find classification head (fc/head/classifier).")

    def get_features(self, x):
        """Extracts features from the penultimate layer (Pre-Logits)."""

        # 1. ResNet / WideResNet
        if isinstance(self.model, ResNet):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            return torch.flatten(x, 1)

        # 2. Vision Transformer (ViT)
        elif isinstance(self.model, VisionTransformer):
            x = self.model._process_input(x)
            n = x.shape[0]

            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.model.encoder(x)
            return x[:, 0]  # Class token

        # 3. Swin Transformer
        elif isinstance(self.model, SwinTransformer):
            # Swin logic: features -> norm -> permute -> avgpool -> flatten
            x = self.model.features(x)
            x = self.model.norm(x)
            x = self.model.permute(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)

        # 4. ConvNeXt (Fixing the Crash)
        elif self.is_convnext:
            x = self.model.features(x)
            x = self.model.avgpool(x)  # (N, C, 1, 1)

            # Apply the specific LayerNorm from the classifier block
            x = self.convnext_norm(x)
            x = self.convnext_flatten(x)  # Flatten to (N, C)
            return x

        # 5. DenseNet
        elif isinstance(self.model, DenseNet):
            features = self.model.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            return torch.flatten(out, 1)

        # Fallback (May fail for unsupported archs)
        return self.model(x)

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature = self.get_features(x)
        logits = self.fc(feature)

        if return_feature:
            return logits, feature
        return logits

    def forward_threshold(self, x, threshold):
        """Required by ReAct."""
        feature = self.get_features(x)
        # Clip features directly
        feature = feature.clip(max=threshold)
        logits = self.fc(feature)
        return logits


def load_backbone(name, device):
    print(f"Loading backbone: {name}...")

    if name == 'resnet50':
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    elif name == 'vit_b_16':
        base = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    elif name == 'swin_t':
        base = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

    elif name == 'convnext_t':
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    elif name == 'densenet121':
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    else:
        raise ValueError(
            f"Backbone {name} not supported. Available: resnet50, vit_b_16, swin_t, convnext_t, densenet121")

    # Wrap for OpenOOD compatibility
    model = OpenOODWrapper(base)
    return model.to(device).eval()