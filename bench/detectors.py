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
    MDSPostprocessor
)


class Config:
    """
    A helper class that recursively converts a nested dictionary into an object
    with dot-notation access.
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

    # Shared Dataset Config
    # Many OpenOOD postprocessors (GradNorm, MDS) look up 'dataset.name' or 'num_classes'.
    # We default to ImageNet (1000 classes).
    base_dataset_cfg = {
        'name': 'imagenet',
        'num_classes': 1000,
        'image_resize': 256,
        'image_crop': 224
    }

    # Helper to construct the full config object
    def build_cfg(postprocessor_args, postprocessor_sweep):
        return {
            'dataset': base_dataset_cfg,
            'postprocessor': {
                'postprocessor_args': postprocessor_args,
                'postprocessor_sweep': postprocessor_sweep
            }
        }

    # 1. MSP (Maximum Softmax Probability)
    if name == 'msp':
        cfg = build_cfg({}, {})
        return BasePostprocessor(Config(cfg))

    # 2. MaxLogit
    elif name == 'maxlogit':
        cfg = build_cfg({}, {})
        return MaxLogitPostprocessor(Config(cfg))

    # 3. ReAct
    elif name == 'react':
        cfg = build_cfg(
            {'percentile': 90},
            {'percentile': [90]}
        )
        return ReactPostprocessor(Config(cfg))

    # 4. ASH (Activation Shaping)
    elif name == 'ash':
        cfg = build_cfg(
            {'percentile': 90},
            {'percentile': [90]}
        )
        return ASHPostprocessor(Config(cfg))

    # 5. DICE (Sparsification)
    elif name == 'dice':
        cfg = build_cfg(
            {'p': 90},
            {'p': [90]}
        )
        return DICEPostprocessor(Config(cfg))

    # 6. KL Matching
    elif name == 'kl_matching':
        cfg = build_cfg({}, {})
        return KLMatchingPostprocessor(Config(cfg))

    # 7. KNN (k-Nearest Neighbors)
    # Note: Requires 'train' loader in setup()
    elif name == 'knn':
        cfg = build_cfg(
            {'K': 50},  # Key must be 'K', not 'nearest_neighbors'
            {'K': [50]}
        )
        return KNNPostprocessor(Config(cfg))

    # 8. GradNorm
    # Note: Explicitly checks config.dataset.name
    elif name == 'gradnorm':
        cfg = build_cfg({}, {})
        pp =  GradNormPostprocessor(Config(cfg))
        pp.requires_grad = True
        return pp


    # 9. EBO (Energy-based Out-of-distribution detection)
    elif name == 'ebo':
        cfg = build_cfg(
            {'temperature': 1.0},
            {'temperature': [1.0]}
        )
        return EBOPostprocessor(Config(cfg))

    # 10. MDS (Mahalanobis Distance Score)
    # Note: Requires 'train' loader in setup() and checks num_classes
    elif name == 'mds':
        cfg = build_cfg({}, {})
        return MDSPostprocessor(Config(cfg))

    else:
        raise ValueError(f"Detector '{name}' not supported.")