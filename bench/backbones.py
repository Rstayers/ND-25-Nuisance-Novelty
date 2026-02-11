# bench/backbones.py

import os
from typing import Any, Dict, Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torchvision.models as tvm

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML is required for ln_dataset config loading. `pip install pyyaml`.") from e

_BACKBONE_TO_CFG_KEY = {
    "resnet50": "resnet_ckpt",
    "vit_b_16": "vit_ckpt",
    "convnext_t": "convnext_ckpt",
    "swin_t": "swin_ckpt",
    "densenet121": "densenet_ckpt",
}
import torch
import torch.nn as nn


class OpenOODTorchvisionAdapter(nn.Module):
    """
    Wrap a torchvision model to provide the OpenOOD net API:
      - forward(x, return_feature=..., return_feature_list=...)
      - get_fc() -> (W, b) as numpy (what OpenOOD postprocessors expect)
      - get_fc_layer() -> nn.Linear
    We capture the penultimate embedding via a forward_pre_hook on the last nn.Linear.
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self._fc = self._find_last_linear(self.base)
        self._last_feat = None

        def _hook(module, inputs):
            feat = inputs[0]
            # keep graph only if grad is enabled
            self._last_feat = feat if torch.is_grad_enabled() else feat.detach()

        self._hook_handle = self._fc.register_forward_pre_hook(_hook)

    @staticmethod
    def _find_last_linear(m: nn.Module) -> nn.Linear:
        last = None
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                last = mod
        if last is None:
            raise RuntimeError("OpenOODTorchvisionAdapter: could not find an nn.Linear classifier head.")
        return last

    def __getattr__(self, name: str):
        # delegate unknown attrs to base (common OpenOOD pattern)
        if name in {"base", "_fc", "_last_feat", "_hook_handle"}:
            return super().__getattr__(name)
        return getattr(self.base, name)

    def get_fc_layer(self) -> nn.Linear:
        return self._fc

    def get_fc(self):
        w = self._fc.weight.detach().cpu().numpy()
        b = None if self._fc.bias is None else self._fc.bias.detach().cpu().numpy()
        return w, b

    def forward(self, x, return_feature: bool = False, return_feature_list: bool = False):
        logits = self.base(x)

        if return_feature or return_feature_list:
            if self._last_feat is None:
                raise RuntimeError("OpenOODTorchvisionAdapter: feature hook did not fire; cannot return features.")
            feat = self._last_feat
            if return_feature_list:
                return logits, [feat]
            return logits, feat

        return logits

    def forward_threshold(self, x: torch.Tensor, threshold: float):
        """
        Performs forward pass with feature clipping (rectification) before the final layer.
        Required by ReAct and ASH detectors.
        """
        # Define a pre-hook that clips the input to the head layer
        def clip_hook(module, args):
            feat = args[0]
            clipped_feat = torch.clamp(feat, max=threshold)
            return (clipped_feat,) + args[1:]

        # Register the pre-hook temporarily
        handle = self._fc.register_forward_pre_hook(clip_hook)

        try:
            logits = self.base(x)
        finally:
            handle.remove()

        return logits


class OpenOODWrapper(nn.Module):
    """
    Wraps a standard torchvision model to support the OpenOOD signature:
    1. forward(x, return_feature=False, return_feature_list=False)
    2. forward_threshold(x, threshold)  <-- Required by ReAct
    """

    def __init__(self, model: nn.Module, backbone_name: str):
        super().__init__()
        self.model = model
        self.backbone_name = backbone_name
        self._captured_feature = None

        # Identify the classification head layer to hook based on architecture
        self.head_layer = None
        if backbone_name == "resnet50":
            self.head_layer = model.fc
        elif backbone_name == "densenet121":
            self.head_layer = model.classifier
        elif backbone_name == "vit_b_16":
            self.head_layer = model.heads.head
        elif backbone_name == "swin_t":
            self.head_layer = model.head
        elif backbone_name == "convnext_t":
            # ConvNeXt classifier is a Sequential; the linear layer is at index 2
            self.head_layer = model.classifier[2]

        if self.head_layer is None:
            raise ValueError(f"Could not identify head layer for backbone: {backbone_name}")

        # Register a hook to capture the input to the classification head (which is the feature)
        # This persists for the life of the wrapper
        self.head_layer.register_forward_hook(self._feature_hook)

    def _feature_hook(self, module, inputs, outputs):
        # inputs is a tuple of args; the feature tensor is the first arg
        self._captured_feature = inputs[0]

    def forward(self, x: torch.Tensor, return_feature: bool = False, return_feature_list: bool = False):
        # Reset captured feature
        self._captured_feature = None

        # Run standard forward pass (hooks will fire)
        logits = self.model(x)

        if return_feature or return_feature_list:
            if self._captured_feature is None:
                raise RuntimeError("Feature was not captured during forward pass. Check hook registration.")

            # Helper to handle return types
            feat = self._captured_feature

            if return_feature_list:
                return logits, [feat]
            return logits, feat

        return logits

    def forward_threshold(self, x: torch.Tensor, threshold: float):
        """
        Performs forward pass with feature clipping (rectification) before the final layer.
        Required by ReAct detector.
        """

        # Define a pre-hook that clips the input to the head layer
        def clip_hook(module, args):
            # args[0] is the feature tensor entering the head
            feat = args[0]
            # Clip (Rectify) the features
            clipped_feat = torch.clamp(feat, max=threshold)
            # Return tuple to replace original args
            return (clipped_feat,) + args[1:]

        # Register the pre-hook temporarily
        handle = self.head_layer.register_forward_pre_hook(clip_hook)

        try:
            # Run the standard forward pass.
            # The hook will intercept the features right before the head, clip them,
            # and pass the clipped features to the head.
            logits = self.model(x)
        finally:
            # Always remove the hook so subsequent normal forward calls aren't affected
            handle.remove()

        return logits

    def __getattr__(self, name: str):
        # Forward attribute access to the underlying model (e.g. model.fc, model.eval)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handles DataParallel/DistributedDataParallel checkpoints.
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    # ckpt can be a raw state_dict or a dict with nested keys
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Might already be a plain state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unrecognized checkpoint format (expected state_dict or dict containing one).")


def _build_torchvision_model(backbone: str, num_classes: int, use_pretrained_imagenet: bool):
    """
    Ensures backbone architecture and weights match generate_ln.py exactly.
    Uses IMAGENET1K_V1 to ensure consistency across the pipeline.
    """
    if backbone == "resnet50":
        if use_pretrained_imagenet:
            # Match generate_ln.py: explicitly use V1 weights and keep 1000-class head
            weights = tvm.ResNet50_Weights.IMAGENET1K_V1
            model = tvm.resnet50(weights=weights)
        else:
            # Match generate_ln.py: init with specific num_classes for custom training
            model = tvm.resnet50(num_classes=num_classes)
        return model

    if backbone == "densenet121":
        if use_pretrained_imagenet:
            weights = tvm.DenseNet121_Weights.IMAGENET1K_V1
            model = tvm.densenet121(weights=weights)
        else:
            model = tvm.densenet121(num_classes=num_classes)
        return model

    if backbone == "vit_b_16":
        if use_pretrained_imagenet:
            weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1
            model = tvm.vit_b_16(weights=weights)
        else:
            model = tvm.vit_b_16(num_classes=num_classes)
        return model

    if backbone == "convnext_t":
        if use_pretrained_imagenet:
            # Note: torchvision names this 'tiny', generation uses 'tiny'
            weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            model = tvm.convnext_tiny(weights=weights)
        else:
            model = tvm.convnext_tiny(num_classes=num_classes)
        return model
    if backbone == "swin_t":
        if use_pretrained_imagenet:
            # Note: torchvision names this 'tiny', generation uses 'tiny'
            weights = tvm.Swin_T_Weights.IMAGENET1K_V1
            model = tvm.swin_t(weights=weights)
        else:
            model = tvm.swin_t(num_classes=num_classes)
        return model

    raise ValueError(f"Unknown backbone: {backbone}")


def load_backbone_from_ln_config(
        backbone: str,
        device: torch.device,
        ln_config_path: str,
) -> torch.nn.Module:
    """
    Load a backbone according to ln_dataset YAML config.
    - If models.use_torchvision: True, load torchvision pretrained (ImageNet weights).
    - Else, load checkpoint path given in the YAML for this backbone.
    """
    cfg = _read_yaml(ln_config_path)

    num_classes = int(cfg["dataset"]["num_classes"])
    use_torchvision = bool(cfg["models"].get("use_torchvision", False))

    # Build base torchvision model
    model = _build_torchvision_model(backbone, num_classes, use_pretrained_imagenet=use_torchvision)

    # If not torchvision weights, load ckpt specified in config
    if not use_torchvision:
        cfg_key = _BACKBONE_TO_CFG_KEY.get(backbone)
        if cfg_key is None:
            raise ValueError(f"No config key mapping for backbone={backbone}")

        ckpt_path = cfg["models"].get(cfg_key, None)
        if ckpt_path is None:
            raise ValueError(f"Config missing models.{cfg_key} in {ln_config_path}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        raw = torch.load(ckpt_path, map_location="cpu")
        state = _extract_state_dict(raw)
        state = _strip_module_prefix(state)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading {ckpt_path}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(
                f"[WARN] Unexpected keys when loading {ckpt_path}: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Wrap model to support OpenOOD forward(x, return_feature=True) and forward_threshold signature
    model = OpenOODTorchvisionAdapter(model)
    return model.to(device).eval()