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


def get_detector(name):
    """
    Factory to return initialized OpenOOD Postprocessors with robust default configs.
    Includes 'dataset' info to satisfy detectors (GradNorm/MDS) that query it.
    """
    name = name.lower()

    base_dataset_cfg = {
        "name": "imagenet",
        "num_classes": 1000,
        "image_resize": 256,
        "image_crop": 224,
    }

    def build_cfg(postprocessor_args, postprocessor_sweep):
        return {
            "dataset": base_dataset_cfg,
            "postprocessor": {
                "postprocessor_args": postprocessor_args,
                "postprocessor_sweep": postprocessor_sweep,
            },
        }

    def _tag(pp):
        # handy for debugging/logging; harmless for OpenOOD
        pp.bench_name = name
        return pp

    if name == "msp":
        cfg = build_cfg({}, {})
        return _tag(BasePostprocessor(Config(cfg)))

    elif name == "maxlogit":
        cfg = build_cfg({}, {})
        return _tag(MaxLogitPostprocessor(Config(cfg)))

    elif name == "react":
        cfg = build_cfg({"percentile": 90}, {"percentile": [90]})
        return _tag(ReactPostprocessor(Config(cfg)))

    elif name == "ash":
        cfg = build_cfg({"percentile": 90}, {"percentile": [90]})
        return _tag(ASHPostprocessor(Config(cfg)))

    elif name == "dice":
        cfg = build_cfg({"p": 90}, {"p": [90]})
        return _tag(DICEPostprocessor(Config(cfg)))

    elif name == "kl_matching":
        cfg = build_cfg({}, {})
        return _tag(KLMatchingPostprocessor(Config(cfg)))

    elif name == "knn":
        cfg = build_cfg({"K": 50}, {"K": [50]})
        return _tag(KNNPostprocessor(Config(cfg)))

    elif name == "gradnorm":
        cfg = build_cfg({}, {})
        return _tag(GradNormPostprocessor(Config(cfg)))

    elif name == "ebo":
        cfg = build_cfg({"temperature": 1.0}, {"temperature": [1.0]})
        return _tag(EBOPostprocessor(Config(cfg)))

    elif name == "mds":
        cfg = build_cfg({}, {})
        return _tag(MDSPostprocessor(Config(cfg)))

    else:
        raise ValueError(f"Detector '{name}' not supported.")
