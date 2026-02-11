# bench/detectors.py - FIXED ODIN and other detectors

import torch
import torch.nn as nn
from typing import Any
import numpy as np
import openood.postprocessors.mds_postprocessor as mds_lib

from openood.postprocessors import (
    BasePostprocessor,
    ReactPostprocessor,
    ASHPostprocessor,
    DICEPostprocessor,
    MaxLogitPostprocessor,
    KLMatchingPostprocessor,
    KNNPostprocessor,
    GradNormPostprocessor,
    EBOPostprocessor,
    MDSPostprocessor,
    VIMPostprocessor,
    SHEPostprocessor
)
import torch.nn.functional as F  # <-- FIX THIS LINE

from bench.datasets import get_dataset_config
from bench.PostMax import PostMaxPostprocessor

_DETECTORS_REQUIRING_TRAIN = {
    "knn", "mds", "dice", "ash", "vim", "she", "postmax", "react"
}


def requires_train_loader(name: str) -> bool:
    return name.lower() in _DETECTORS_REQUIRING_TRAIN


class Config:
    """Recursively converts a nested dict into an object with dot-notation access."""

    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


# =============================================================================
# FIXED ODIN - Custom implementation that actually works
# =============================================================================

class FixedODINPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        args = config.postprocessor.postprocessor_args
        self.temperature = getattr(args, 'temperature', 1000)
        self.noise = getattr(args, 'noise', 0.0014)

    @torch.enable_grad()
    def postprocess(self, net, data):
        data = data.clone().requires_grad_(True)

        output = net(data)

        # Temperature scaling for gradient computation only
        output_scaled = output / self.temperature
        labels = output.argmax(dim=1)

        loss = F.cross_entropy(output_scaled, labels)
        loss.backward()

        # Gradient perturbation
        gradient = torch.sign(data.grad.data)

        # Normalize by ImageNet std
        gradient[:, 0] /= 0.229
        gradient[:, 1] /= 0.224
        gradient[:, 2] /= 0.225

        perturbed = data.detach() - self.noise * gradient

        # Final forward WITHOUT temperature scaling
        with torch.no_grad():
            output_final = net(perturbed)
            conf, pred = F.softmax(output_final, dim=1).max(dim=1)

        return pred, conf


# =============================================================================
# FIXED ReAct - Ensure percentile is computed correctly
# =============================================================================

class FixedReActPostprocessor(BasePostprocessor):
    """ReAct with proper percentile computation."""

    def __init__(self, config):
        super().__init__(config)
        args = config.postprocessor.postprocessor_args
        self.percentile = getattr(args, 'percentile', 90)
        self.threshold = None

    def setup(self, net, id_loader_dict, ood_loader_dict):
        net.eval()

        activation_list = []
        loader = id_loader_dict.get('val') or id_loader_dict.get('train')

        if loader is None:
            raise ValueError("ReAct setup requires 'val' or 'train' loader")

        print(f"ReAct setup: computing {self.percentile}th percentile...")
        with torch.no_grad():
            for batch in loader:
                data = batch['data'].cuda()
                _, features = net(data, return_feature=True)
                activation_list.append(features.cpu())

        all_activations = torch.cat(activation_list, dim=0)
        self.threshold = np.percentile(all_activations.numpy(), self.percentile)
        print(f"ReAct threshold: {self.threshold:.4f}")

    def postprocess(self, net, data):
        if self.threshold is None:
            raise RuntimeError("ReAct: setup() must be called before postprocess()")

        with torch.no_grad():
            logits = net.forward_threshold(data, self.threshold)
            conf, pred = F.softmax(logits, dim=1).max(dim=1)

        return pred, conf


class FixedASHPostprocessor(BasePostprocessor):
    """ASH with per-sample activation pruning."""

    def __init__(self, config):
        super().__init__(config)
        args = config.postprocessor.postprocessor_args
        self.percentile = getattr(args, 'percentile', 90)

    def setup(self, net, id_loader_dict, ood_loader_dict):
        pass  # ASH computes threshold per-sample

    def postprocess(self, net, data):
        with torch.no_grad():
            logits, features = net(data, return_feature=True)

            # Compute per-sample threshold (percentile along feature dim)
            features_np = features.cpu().numpy()
            threshold_np = np.percentile(features_np, self.percentile, axis=1, keepdims=True)
            threshold = torch.from_numpy(threshold_np).to(device=features.device, dtype=features.dtype)

            # ASH-P: Zero activations below threshold
            features_ash = features * (features >= threshold).float()

            # Recompute logits with pruned features
            fc = net.get_fc_layer()
            logits_ash = fc(features_ash)

            conf, pred = F.softmax(logits_ash, dim=1).max(dim=1)

        return pred, conf


# =============================================================================
# FACTORY
# =============================================================================

def get_detector(name, dataset_name="ImageNet-Val"):
    """Factory for OpenOOD Postprocessors."""
    name = name.lower()

    ds_cfg = get_dataset_config(dataset_name)
    num_classes = ds_cfg.get("num_classes", 1000)

    if dataset_name not in mds_lib.num_classes_dict:
        mds_lib.num_classes_dict[dataset_name] = num_classes

    if "imagenet" in dataset_name.lower():
        ood_dataset_name = "imagenet"
    elif "cifar10" in dataset_name.lower() and "cifar100" not in dataset_name.lower():
        ood_dataset_name = "cifar10"
    elif "cifar100" in dataset_name.lower():
        ood_dataset_name = "cifar100"
    else:
        ood_dataset_name = dataset_name

    base_dataset_cfg = {
        "name": ood_dataset_name,
        "num_classes": num_classes,
        "num_classes_id": num_classes,
        "image_size": 224,
        "interpolation": "bilinear",
        "normalization_type": "imagenet"
    }

    def build_cfg(detector_hyperparams, search_space=None):
        return {
            "dataset": base_dataset_cfg,
            "postprocessor": {
                "name": name,
                "postprocessor_args": detector_hyperparams,
                "postprocessor_sweep": search_space or {}
            }
        }

    def _tag(pp):
        pp.bench_name = name
        return pp

    # --- Standard detectors ---
    if name == "msp":
        return _tag(BasePostprocessor(Config(build_cfg({}))))

    elif name == "maxlogit":
        return _tag(MaxLogitPostprocessor(Config(build_cfg({}))))

    elif name == "ebo":
        return _tag(EBOPostprocessor(Config(build_cfg({"temperature": 1}))))

    # --- Fixed detectors ---
    elif name == "odin":
        return _tag(FixedODINPostprocessor(Config(build_cfg({
            "temperature": 100,  # Lower from 1000
            "noise": 0.001  # Lower from 0.0014
        }))))
    elif name == "react":
        return _tag(FixedReActPostprocessor(Config(build_cfg({"percentile": 90}))))

    elif name == "ash":
        return _tag(FixedASHPostprocessor(Config(build_cfg({"percentile": 90}))))

    # --- Detectors requiring training data ---
    elif name == "dice":
        return _tag(DICEPostprocessor(Config(build_cfg({"p": 90}))))

    elif name == "knn":
        return _tag(KNNPostprocessor(Config(build_cfg({"K": 50}))))

    elif name == "mds":
        return _tag(MDSPostprocessor(Config(build_cfg({}))))

    elif name == "vim":
        return _tag(VIMPostprocessor(Config(build_cfg({"dim": 512}))))

    elif name == "she":
        return _tag(SHEPostprocessor(Config(build_cfg({"metric": "inner_product"}))))

    elif name == "postmax":
        return _tag(PostMaxPostprocessor(Config(build_cfg({}))))

    else:
        raise ValueError(f"Unknown detector: {name}")