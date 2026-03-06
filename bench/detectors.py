# bench/detectors.py - Streamlined detector factory using stock OpenOOD

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import numpy as np
import openood.postprocessors.mds_postprocessor as mds_lib

from openood.postprocessors import (
    BasePostprocessor,
    ODINPostprocessor,
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


class FixedODINPostprocessor(BasePostprocessor):
    """
    Fixed ODIN implementation that properly handles gradients.

    ODIN: Out-of-Distribution Detector for Neural Networks
    Uses temperature scaling and input perturbation.
    """

    def __init__(self, config):
        super().__init__(config)
        # Config objects don't support .get(), use getattr instead
        args = config.postprocessor.postprocessor_args
        self.temperature = getattr(args, "temperature", 1000)
        self.noise = getattr(args, "noise", 0.0014)
        # ImageNet normalization std for gradient scaling
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def postprocess(self, net, data):
        """
        Compute ODIN score with temperature scaling and input perturbation.
        """
        device = data.device
        self.std = self.std.to(device)

        # Enable gradient computation for input perturbation
        data = data.clone().requires_grad_(True)

        # Forward pass with temperature scaling
        logits = net(data)
        logits_scaled = logits / self.temperature

        # Get pseudo-labels (max predictions)
        pseudo_labels = logits_scaled.argmax(dim=1)

        # Compute cross-entropy loss for perturbation direction
        loss = F.cross_entropy(logits_scaled, pseudo_labels)

        # Backward pass to get input gradients
        loss.backward()

        # Compute perturbation: epsilon * sign(gradient), normalized by std
        gradient = data.grad.data
        gradient_normalized = gradient / self.std
        perturbation = self.noise * torch.sign(gradient_normalized)

        # Apply perturbation
        data_perturbed = data.detach() - perturbation

        # Forward pass on perturbed input
        with torch.no_grad():
            logits_perturbed = net(data_perturbed)
            logits_perturbed_scaled = logits_perturbed / self.temperature
            softmax_scores = F.softmax(logits_perturbed_scaled, dim=1)
            conf, pred = torch.max(softmax_scores, dim=1)

        return pred, conf

from bench.datasets import get_dataset_config
from bench.PostMax import PostMaxPostprocessor
from bench.COSTARR import COSTARR
from tqdm import tqdm

_DETECTORS_REQUIRING_TRAIN = {
    "knn", "mds", "dice", "ash", "vim", "she", "postmax", "react", "costarr"
}


class COSTARRPostprocessor(BasePostprocessor):
    """Wrapper for COSTARR detector to match OpenOOD BasePostprocessor API."""

    def __init__(self, config):
        super().__init__(config)
        self.costarr_model = None
        self.weights = None

    def setup(self, net, id_loader_dict, ood_loader_dict):
        """Extract features from training data and fit COSTARR model."""
        # Get classification head weights from network
        self.weights = self._get_fc_weights(net)

        # Extract (logits, features) from training data
        train_loader = id_loader_dict.get("train") or id_loader_dict.get("val")
        logits, features, labels = self._extract_features(net, train_loader)

        # Filter to correctly classified samples
        preds = logits.argmax(dim=1)
        correct_mask = (preds == labels)

        print(f"[COSTARR] Using {correct_mask.sum()}/{len(labels)} correctly classified samples for setup")

        # Initialize COSTARR model
        self.costarr_model = COSTARR(
            logits[correct_mask],
            features[correct_mask],
            self.weights
        )

    def _get_fc_weights(self, net):
        """Extract classification head weights from network."""
        # Try common attribute names for the classification head
        if hasattr(net, 'fc'):
            return net.fc.weight.detach().clone()
        elif hasattr(net, 'head'):
            if hasattr(net.head, 'weight'):
                return net.head.weight.detach().clone()
            elif hasattr(net.head, 'fc'):
                return net.head.fc.weight.detach().clone()
        elif hasattr(net, 'classifier'):
            if isinstance(net.classifier, nn.Linear):
                return net.classifier.weight.detach().clone()
            elif isinstance(net.classifier, nn.Sequential):
                # Get last linear layer
                for layer in reversed(net.classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.weight.detach().clone()
        elif hasattr(net, 'get_fc'):
            W, _ = net.get_fc()
            return torch.from_numpy(W)

        raise ValueError("Could not find classification head weights in network")

    def _extract_features(self, net, loader):
        """Extract logits, features, and labels from data loader."""
        net.eval()
        device = next(net.parameters()).device

        all_logits = []
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="[COSTARR] Extracting features"):
                if isinstance(batch, dict):
                    data = batch['data'].to(device)
                    labels = batch['label']
                else:
                    data, labels = batch
                    data = data.to(device)

                # Forward pass with features
                logits, features = net(data, return_feature=True)

                all_logits.append(logits.cpu())
                all_features.append(features.cpu())
                all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))

        return (
            torch.cat(all_logits, dim=0),
            torch.cat(all_features, dim=0),
            torch.cat(all_labels, dim=0)
        )

    def postprocess(self, net, data):
        """Compute COSTARR scores."""
        # Get logits and features
        logits, features = net(data, return_feature=True)

        # Get COSTARR rescored outputs
        rescored = self.costarr_model.ReScore(logits.cpu(), features.cpu())

        # Return (prediction, confidence)
        conf, pred = torch.max(rescored, dim=1)
        return pred, conf


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

    # --- Standard detectors (stock OpenOOD) ---
    if name == "msp":
        return _tag(BasePostprocessor(Config(build_cfg({}))))

    elif name == "maxlogit":
        return _tag(MaxLogitPostprocessor(Config(build_cfg({}))))

    elif name == "ebo":
        return _tag(EBOPostprocessor(Config(build_cfg({"temperature": 1}))))

    elif name == "odin":
        return _tag(FixedODINPostprocessor(Config(build_cfg({
            "temperature": 1000,
            "noise": 0.0014
        }))))

    elif name == "react":
        return _tag(ReactPostprocessor(Config(build_cfg({"percentile": 90}))))

    elif name == "ash":
        return _tag(ASHPostprocessor(Config(build_cfg({"percentile": 90}))))

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

    elif name == "costarr":
        return _tag(COSTARRPostprocessor(Config(build_cfg({}))))

    else:
        raise ValueError(f"Unknown detector: {name}")
