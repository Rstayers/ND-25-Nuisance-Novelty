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
        if isinstance(model, ConvNeXt):
            # ConvNeXt.classifier = Sequential(LayerNorm2d, Flatten, Linear)
            # We treat the Linear layer as the 'fc' head.
            # We must treat LayerNorm + Flatten as part of the feature extractor.
            self.fc = model.classifier[2]  # The Linear Layer (768 -> 1000)
            self.convnext_norm = model.classifier[0]  # LayerNorm2d
            self.convnext_flatten = model.classifier[1]  # Flatten
            self.is_convnext = True
        else:
            self.is_convnext = False

            # --- 2. Locate the Linear Layer (fc) ---
            if hasattr(model, 'fc'):
                self.fc = model.fc
            elif hasattr(model, 'heads'):
                # ViT: self.heads is usually a Sequential containing the Linear layer
                if isinstance(model.heads, nn.Sequential):
                    self.fc = model.heads[-1]
                else:
                    self.fc = model.heads
            elif hasattr(model, 'classifier'):
                # DenseNet, VGG: classifier is often Sequential
                if isinstance(model.classifier, nn.Sequential):
                    self.fc = model.classifier[-1]
                else:
                    self.fc = model.classifier
            elif hasattr(model, 'head'):
                # Swin
                self.fc = model.head
            else:
                raise ValueError(f"Could not find classification head for {type(model)}.")

    def get_fc(self):
        """
        Required by GradNorm.
        Returns weights/bias as NumPy arrays, not Tensors.
        """
        w = self.fc.weight.detach().cpu().numpy()
        if self.fc.bias is not None:
            b = self.fc.bias.detach().cpu().numpy()
        else:
            b = None
        return w, b

    def get_feature_dim(self):
        """
        Returns the input dimension of the final fully connected layer.
        """
        return self.fc.in_features

    def get_features(self, x):
        """
        Extracts the penultimate feature vector (before the final linear layer).
        """
        # --- A. ConvNeXt ---
        if self.is_convnext:
            x = self.model.features(x)
            # CRITICAL FIX: Add the missing Global Average Pooling
            x = self.model.avgpool(x)
            x = self.convnext_norm(x)
            x = self.convnext_flatten(x)
            return x

        # --- B. Vision Transformer (ViT) ---
        if isinstance(self.model, VisionTransformer):
            # Manual forward pass to stop before heads
            x = self.model._process_input(x)
            n = x.shape[0]
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.model.encoder(x)
            x = x[:, 0]  # Take CLS token
            return x

        # --- C. Swin Transformer ---
        if isinstance(self.model, SwinTransformer):
            x = self.model.features(x)
            x = self.model.norm(x)
            x = self.model.permute(x)
            x = self.model.avgpool(x)
            x = self.model.flatten(x)
            return x

        # --- D. ResNet / DenseNet / Others ---
        # Try newer torchvision 'forward_features' if available
        if hasattr(self.model, 'forward_features'):
            out = self.model.forward_features(x)
        else:
            # Manual extraction for older ResNet/DenseNet
            if isinstance(self.model, ResNet):
                out = self.model.conv1(x)
                out = self.model.bn1(out)
                out = self.model.relu(out)
                out = self.model.maxpool(out)
                out = self.model.layer1(out)
                out = self.model.layer2(out)
                out = self.model.layer3(out)
                out = self.model.layer4(out)
                out = self.model.avgpool(out)
            elif isinstance(self.model, DenseNet):
                out = self.model.features(x)
                out = F.relu(out, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
            else:
                # Fallback
                if hasattr(self.model, 'features'):
                    out = self.model.features(x)
                    if hasattr(self.model, 'avgpool'):
                        out = self.model.avgpool(out)
                else:
                    raise NotImplementedError(f"Feature extraction not implemented for {type(self.model)}")

        # Flatten 4D outputs (e.g. ResNet) to 2D
        if out.dim() == 4:
            out = torch.flatten(out, 1)

        # Handle ViT-like outputs if they slipped through logic above
        if out.dim() == 3:
            return out[:, 0]

        return out

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature = self.get_features(x)
        logits = self.fc(feature)

        if return_feature:
            return logits, feature
        return logits

    def forward_threshold(self, x, threshold):
        """Required by ReAct."""
        feature = self.get_features(x)
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
        raise ValueError(f"Backbone {name} not supported.")

    model = OpenOODWrapper(base)
    model.to(device)
    model.eval()
    return model