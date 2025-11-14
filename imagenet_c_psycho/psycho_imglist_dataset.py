import os
import torch
from torch.utils.data import Dataset
from openood.datasets.imglist_dataset import ImglistDataset


class ImagenetCPsychoDataset(ImglistDataset):
    """
    Extension of OpenOOD's ImglistDataset for ImageNet-C with psychophysical weighting.
    Each sample returns (image, label, perceptual_value) where perceptual_value is a
    scalar in [0, 1] derived from corruption severity (1–5).
    """

    def __init__(self, config, split_name='train', **kwargs):
        super().__init__(config, split_name=split_name, **kwargs)

        # Load psychophysical weighting parameters
        self.use_weights = config.get("use_perceptual_weights", True)
        self.severity_weight_mode = config.get("severity_weight_mode", "inverse")
        self.included_severities = config.get("included_severities", [1, 2, 3])
        self.normalize_weights = config.get("normalize_weights", True)

        # Mapping for explicit weights (reaction-time inspired)
        self.perceptual_weight_mapping = config.get("perceptual_weight_mapping", {
            1: 1.0,
            2: 0.85,
            3: 0.65,
            4: 0.45,
            5: 0.25
        })

        # Collect list of image paths and labels
        self._filter_by_severity()

        # Compute normalization if needed
        if self.normalize_weights:
            weights = torch.tensor(list(self.perceptual_weight_mapping.values()), dtype=torch.float32)
            self.min_w = float(weights.min())
            self.max_w = float(weights.max())
        else:
            self.min_w = self.max_w = None

    def _filter_by_severity(self):
        """
        Optionally filter dataset entries by severity level using folder names
        like .../gaussian_noise/3/image.png.
        """
        filtered_data = []
        for item in self.data:
            path = item["img_path"]
            try:
                severity = int(os.path.normpath(path).split(os.sep)[-2])
            except ValueError:
                # If not an ImageNet-C corruption path, assume severity=1
                severity = 1
            if severity in self.included_severities:
                filtered_data.append(item)
        self.data = filtered_data

    def _severity_to_weight(self, severity):
        """
        Convert corruption severity (1–5) to a perceptual weight scalar.
        """
        if not self.use_weights:
            return 1.0

        # Direct mapping if available
        if severity in self.perceptual_weight_mapping:
            weight = self.perceptual_weight_mapping[severity]
        else:
            # Fallback: simple inverse or linear scaling
            if self.severity_weight_mode == "inverse":
                weight = 1.0 / severity
            elif self.severity_weight_mode == "linear":
                weight = 1.0 - (severity - 1) / 4.0
            else:
                weight = 1.0

        # Normalize to [0, 1] if enabled
        if self.normalize_weights and self.max_w > self.min_w:
            weight = (weight - self.min_w) / (self.max_w - self.min_w)

        return float(weight)

    def __getitem__(self, idx):
        """
        Returns: image, label, perceptual_weight
        """
        img, label = super().__getitem__(idx)
        path = self.data[idx]["img_path"]

        try:
            severity = int(os.path.normpath(path).split(os.sep)[-2])
        except ValueError:
            severity = 1

        perceptual_value = self._severity_to_weight(severity)
        return img, label, torch.tensor(perceptual_value, dtype=torch.float32)
