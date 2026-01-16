import os
import torch
import torch.nn as nn
from torchvision import models
from bench.datasets import get_dataset_config


# --- WRAPPER FOR FEATURE EXTRACTION ---
class OpenOODWrapper(nn.Module):
    """
    Wraps a standard torchvision model to support:
    logits, features = model(x, return_feature=True)

    It uses a forward hook to capture the input to the final linear layer.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feature = None

        # Identify the final linear layer to hook into
        if hasattr(model, "fc"):
            self.last_layer = model.fc
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                # ConvNeXt: classifier = [LayerNorm, Flatten, Linear]
                self.last_layer = model.classifier[2]
            else:
                # DenseNet: classifier = Linear
                self.last_layer = model.classifier
        elif hasattr(model, "heads"):
            # ViT: heads = Sequential(head=Linear)
            if hasattr(model.heads, "head"):
                self.last_layer = model.heads.head
            else:
                self.last_layer = model.heads
        elif hasattr(model, "head"):
            # Swin: head = Linear
            self.last_layer = model.head
        else:
            # Fallback (might fail for some custom archs)
            raise ValueError("OpenOODWrapper: Could not find final linear layer (fc/classifier/head).")

        # Register the hook
        self.last_layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # input is a tuple of (features, ), we want the tensor
        self.feature = input[0]

    def forward(self, x, return_feature=False):
        # Run standard forward pass (ignoring return_feature arg for the base model)
        logits = self.model(x)

        if return_feature:
            return logits, self.feature
        return logits


# --- CONFIGURATION ---
CARS_CHECKPOINTS = {
    "resnet50": "ln_dataset/assets/cars_models/resnet50_cars.pth",
    "vit_b_16": "ln_dataset/assets/cars_models/vit_b_16_cars.pth",
    "densenet121": "ln_dataset/assets/cars_models/densenet121_cars.pth",
    "convnext_t": "ln_dataset/assets/cars_models/convnext_t_cars.pth",
    "swin_t": "ln_dataset/assets/cars_models/swin_t_cars.pth",
}


def load_backbone(name, device, dataset_name=None):
    """
    Args:
        name: backbone name (e.g., 'resnet50')
        device: torch device
        dataset_name: (str) used to lookup config (num_classes, is_imagenet)
    """
    print(f"Loading Backbone: {name} (Dataset: {dataset_name})")

    # 1. Get Config
    ds_config = get_dataset_config(dataset_name) if dataset_name else {}
    use_torchvision = ds_config.get("is_imagenet", True)
    num_classes = ds_config.get("num_classes", 1000)

    model = None

    # 2. Instantiate Base Model
    if use_torchvision:
        # --- PATH A: Standard ImageNet (Torchvision Weights) ---
        if name == "resnet50":
            model = models.resnet50(weights='IMAGENET1K_V1')
        elif name == "vit_b_16":
            model = models.vit_b_16(weights='IMAGENET1K_V1')
        elif name in ["convnext_t", "convnext_tiny"]:
            model = models.convnext_tiny(weights='IMAGENET1K_V1')
        elif name in ["densenet121", "densenet"]:
            model = models.densenet121(weights='IMAGENET1K_V1')
        elif name in ["swin_t", "swin"]:
            model = models.swin_t(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Unknown backbone: {name}")

    else:
        # --- PATH B: Custom Dataset (Random Init + Checkpoint Load) ---
        if name == "resnet50":
            model = models.resnet50(num_classes=num_classes)
        elif name == "vit_b_16":
            model = models.vit_b_16(num_classes=num_classes)
        elif name in ["convnext_t", "convnext_tiny"]:
            model = models.convnext_tiny(num_classes=num_classes)
        elif name in ["densenet121", "densenet"]:
            model = models.densenet121(num_classes=num_classes)
        elif name in ["swin_t", "swin"]:
            model = models.swin_t(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown backbone: {name}")

        # Load Weights
        ckpt_path = None
        if dataset_name == "Stanford-Cars":
            ckpt_path = CARS_CHECKPOINTS.get(name)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"  --> Loading Custom Weights: {ckpt_path}")
            state = torch.load(ckpt_path, map_location='cpu')

            if 'state_dict' in state:
                state = state['state_dict']

            new_state = {}
            for k, v in state.items():
                k = k.replace("module.", "")
                new_state[k] = v

            try:
                model.load_state_dict(new_state, strict=True)
            except RuntimeError as e:
                print(f"  !! Strict load failed (Retrying strict=False): {e}")
                model.load_state_dict(new_state, strict=False)
        else:
            if not use_torchvision:
                print(f"  !! WARNING: No checkpoint found for {name} on {dataset_name}. Model has random weights!")

    model.to(device)
    model.eval()

    # 3. Wrap Model for OpenOOD Compatibility
    # This wrapper enables `return_feature=True` which your bench script needs
    wrapped_model = OpenOODWrapper(model)
    wrapped_model.to(device)

    return wrapped_model