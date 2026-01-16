# detectors.py
import torch
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
)

from bench.datasets import get_dataset_config

# Only these require ImageNet-Train during setup() in your current zoo.
_DETECTORS_REQUIRING_TRAIN = {"knn", "mds", "dice", "ash"}


def requires_train_loader(name: str) -> bool:
    return name.lower() in _DETECTORS_REQUIRING_TRAIN


class Config:
    """
    Recursively converts a nested dict into an object with dot-notation access.
    """

    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def get_detector(name, dataset_name="ImageNet-Val"):
    """
    Factory for OpenOOD Postprocessors.
    Fetches num_classes dynamically based on the dataset being benchmarked.
    """
    name = name.lower()

    # Fetch Dataset Config
    ds_cfg = get_dataset_config(dataset_name)
    num_classes = ds_cfg.get("num_classes", 1000)

    # Base Config passed to OpenOOD
    base_dataset_cfg = {
        "name": dataset_name,
        "num_classes": num_classes,
        "num_classes_id": num_classes,
        "image_size": 224,
        # OpenOOD sometimes requires these for OOD setup,
        # but for postprocessors we mostly care about num_classes
        "interpolation": "bilinear",
        "normalization_type": "imagenet"
    }

    def build_cfg(detector_hyperparams, search_space):
        return {
            "dataset": base_dataset_cfg,
            "postprocessor": {
                "name": name,
                "postprocessor_args": detector_hyperparams,
                "postprocessor_sweep": search_space
            }
        }

    def _tag(pp):
        pp.bench_name = name
        return pp

    if name == "msp":
        return _tag(BasePostprocessor(Config(build_cfg({}, {}))))
    elif name == "maxlogit":
        return _tag(MaxLogitPostprocessor(Config(build_cfg({}, {}))))
    elif name == "react":
        return _tag(ReactPostprocessor(Config(build_cfg({"percentile": 90}, {"percentile": [90]}))))
    elif name == "ash":
        return _tag(ASHPostprocessor(Config(build_cfg({"percentile": 90}, {"percentile": [90]}))))
    elif name == "dice":
        return _tag(DICEPostprocessor(Config(build_cfg({"p": 90}, {"p": [90]}))))
    elif name == "kl_matching":
        return _tag(KLMatchingPostprocessor(Config(build_cfg({}, {}))))
    elif name == "knn":
        return _tag(KNNPostprocessor(Config(build_cfg({"K": 50}, {"K": [50]}))))
    elif name == "gradnorm":
        return _tag(GradNormPostprocessor(Config(build_cfg({}, {}))))
    elif name == "ebo":
        return _tag(EBOPostprocessor(Config(build_cfg({"temperature": 1}, {"temperature": [1]}))))
    elif name == "mds":
        return _tag(MDSPostprocessor(Config(build_cfg({}, {}))))
    else:
        raise ValueError(f"Unknown detector: {name}")
