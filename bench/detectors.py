import torch
from openood.postprocessors import (
    BasePostprocessor,
    ReactPostprocessor,
    ASHPostprocessor,
    DICEPostprocessor,
    MaxLogitPostprocessor,
    KLMatchingPostprocessor
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
    """
    name = name.lower()

    # 1. MSP (Maximum Softmax Probability)
    if name == 'msp':
        cfg = {
            'postprocessor': {
                'postprocessor_args': {},
                'postprocessor_sweep': {}
            }
        }
        return BasePostprocessor(Config(cfg))

    # 2. MaxLogit
    elif name == 'maxlogit':
        cfg = {
            'postprocessor': {
                'postprocessor_args': {},
                'postprocessor_sweep': {}
            }
        }
        return MaxLogitPostprocessor(Config(cfg))

    # 3. ReAct (Rectified Activation)
    elif name == 'react':
        cfg = {
            'postprocessor': {
                'postprocessor_args': {
                    'percentile': 90
                },
                # OpenOOD requires this field to exist even if unused
                'postprocessor_sweep': {
                    'percentile': [90]
                }
            }
        }
        return ReactPostprocessor(Config(cfg))

    # 4. ASH (Activation Shaping)
    elif name == 'ash':
        cfg = {
            'postprocessor': {
                'postprocessor_args': {
                    'percentile': 90
                },
                'postprocessor_sweep': {
                    'percentile': [90]
                }
            }
        }
        return ASHPostprocessor(Config(cfg))

    # 5. DICE (Sparsification)
    elif name == 'dice':
        cfg = {
            'postprocessor': {
                'postprocessor_args': {
                    'p': 90
                },
                'postprocessor_sweep': {
                    'p': [90]
                }
            }
        }
        return DICEPostprocessor(Config(cfg))

    # 6. KL Matching
    elif name == 'kl_matching':
        cfg = {
            'postprocessor': {
                'postprocessor_args': {},
                'postprocessor_sweep': {}
            }
        }
        return KLMatchingPostprocessor(Config(cfg))

    else:
        raise ValueError(f"Detector '{name}' not supported in bench/detectors.py")