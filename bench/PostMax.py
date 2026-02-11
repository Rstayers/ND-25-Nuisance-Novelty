from typing import Any

import torch
import numpy as np
import scipy.stats
from openood.postprocessors import BasePostprocessor
from torch import nn
from tqdm import tqdm


class PostMax:
    def __init__(self, norm: int = 2):
        """
        PostMax scoring using Generalized Pareto Distribution (GPD).

        Parameters
        ----------
        norm : int
            Norm type (e.g. 1, 2, or np.inf) for feature normalization.
        """
        self.norm = norm

    def score(self, model: dict, norm_logits: np.ndarray):
        """
        Compute probabilities from normalized logits using a fitted GPD.

        Parameters
        ----------
        model : dict
            GPD parameters {'shape', 'loc', 'scale'}.
        norm_logits : np.ndarray
            Normalized logits for samples.

        Returns
        -------
        torch.Tensor
            Probabilities mapped via GPD CDF.
        """
        probs = scipy.stats.genpareto.cdf(norm_logits, c=model['shape'], loc=model['loc'], scale=model['scale'])
        return torch.from_numpy(probs)

    def train(self, labels: torch.Tensor, features: torch.Tensor, logits: torch.Tensor):
        """
        Fit GPD to normalized logits.

        Parameters
        ----------
        labels : torch.Tensor
            Ground-truth labels [N].
        features : torch.Tensor
            Feature embeddings [N, D].
        logits : torch.Tensor
            Class logits [N, C].

        Returns
        -------
        dict
            Fitted GPD parameters.
        """
        assert labels.shape[0] == features.shape[0] == logits.shape[0], "Tensors must have the same batch dimension."

        mask = labels == torch.argmax(logits, dim=1)
        filt_labels = labels[mask]
        filt_features = features[mask]
        filt_logits = logits[mask]

        norm_logits = []
        for cls_id in range(filt_logits.shape[1]):
            cls_features = filt_features[filt_labels == cls_id]
            cls_logits = filt_logits[filt_labels == cls_id]

            if cls_features.numel() == 0:
                continue

            max_cls_logits = cls_logits[:, cls_id]
            norm_cls_logits = max_cls_logits / torch.norm(cls_features, p=self.norm, dim=1)
            norm_logits.append(norm_cls_logits)

        norm_logits = torch.cat(norm_logits, dim=0).numpy()
        shape, loc, scale = scipy.stats.genpareto.fit(norm_logits)

        return {'shape': shape, 'loc': loc, 'scale': scale}

    def evaluate(self, model: dict, labels: torch.Tensor, features: torch.Tensor, logits: torch.Tensor, pct: float = 1.0):
        """
        Evaluate model and compute scores.

        Parameters
        ----------
        model : dict
            GPD parameters.
        labels : torch.Tensor
            Ground-truth labels [N].
        features : torch.Tensor
            Feature embeddings [N, D].
        logits : torch.Tensor
            Class logits [N, C].
        pct : float, default=1.0
            Fraction of dataset to evaluate (e.g. 0.5 for 50%).

        Returns
        -------
        torch.Tensor
            Tensor of shape [M, 3] with (label, prediction, probability).
        """
        assert labels.shape[0] == features.shape[0] == logits.shape[0], "Tensors must have the same batch dimension."

        if pct < 1.0:
            num_imgs = int(labels.shape[0] * pct)
            labels, features, logits = labels[:num_imgs], features[:num_imgs], logits[:num_imgs]

        max_logits, preds = torch.max(logits, dim=1)
        norm_logits = max_logits / torch.norm(features, p=self.norm, dim=1)
        probs = self.score(model, norm_logits.numpy())

        # Stack results into [N, 3]
        results = torch.stack((labels, preds, probs), dim=1)

        return results


class PostMaxPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        # Initialize PostMax with norm=2 as per your default
        self.postmax = PostMax(norm=2)
        self.params = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """
        Fit the PostMax GPD model using ID training data.
        """
        net.eval()
        # PostMax needs training data to fit the GPD parameters
        train_loader = id_loader_dict['train']

        features_list = []
        logits_list = []
        labels_list = []

        print("Setting up PostMax...")
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="PostMax Setup"):
                data = batch['data'].cuda()
                labels = batch['label'].cuda()

                # OpenOOD standard: return_feature=True returns (logits, features)
                logits, features = net(data, return_feature=True)

                features_list.append(features.cpu())
                logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())

        # Concatenate all batches
        features = torch.cat(features_list)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Call the user-provided PostMax.train method
        self.params = self.postmax.train(labels, features, logits)
        print("PostMax Setup Complete.")

    def postprocess(self, net: nn.Module, data: Any):
        """
        Inference time: Compute PostMax scores.
        """
        net.eval()
        with torch.no_grad():
            logits, features = net(data, return_feature=True)

            # Replicate PostMax.evaluate logic to prepare inputs
            max_logits, preds = torch.max(logits, dim=1)

            # Normalize logits by feature norm (p=2)
            norm_logits = max_logits / torch.norm(features, p=self.postmax.norm, dim=1)

            # Get probabilities from GPD
            # Note: score returns probabilities, which serve as the confidence score here
            conf = self.postmax.score(self.params, norm_logits.cpu().numpy())

            return preds, conf