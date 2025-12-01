# psycho_bench/psychophysics_loss.py
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PsychWeightedCELoss(nn.Module):
    """
    Dulay-style psychophysical cross-entropy:
      - Heavier penalty for errors on 'easy' samples (low difficulty).
      - Lighter penalty on hard samples.
    Here difficulty ∈ [0,1] comes from ImageNet-C severity.
    """

    def __init__(self, lambda_easy: float = 1.0):
        """
        lambda_easy controls how much more strongly we weight easy samples.
        """
        super().__init__()
        self.lambda_easy = lambda_easy

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: (B, C)
        targets: (B,)
        difficulty: (B,) in [0,1] where 0=easiest, 1=hardest
        """
        ce = F.cross_entropy(logits, targets, reduction="none")
        # Weighting: easy samples (difficulty small) get larger weight.
        # w_i = 1 + λ * (1 - difficulty_i)
        weights = 1.0 + self.lambda_easy * (1.0 - difficulty)
        return (weights * ce).mean()


class ExitIndexLoss(nn.Module):
    """
    MSDNet-style exit loss:
      L_exit = |E_target(x) - E_pred(x)|
    where:
      - E_target(x) is derived from severity / synthetic RT,
      - E_pred(x) is predicted exit index based on per-exit confidences.
    """

    def __init__(self, n_exits: int, p_conf: float = 1.0):
        """
        n_exits: number of MSDNet exits.
        p_conf: exponent to emphasize confidence when computing E_pred.
        """
        super().__init__()
        self.n_exits = n_exits
        self.p_conf = p_conf

    def forward(
        self,
        logits_list: Sequence[torch.Tensor],
        targets: torch.Tensor,
        exit_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits_list: list of Tensors [ (B, C), ... ] for each exit.
        targets: (B,) class labels (unused here but kept for symmetry).
        exit_targets: (B,) integer exit indices ∈ {0,...,n_exits-1}.
        """
        # Compute per-exit predicted confidences (max softmax prob per exit)
        confs_per_exit = []
        for logits in logits_list:
            probs = F.softmax(logits, dim=1)
            confs, _ = probs.max(dim=1)  # (B,)
            confs_per_exit.append(confs)

        # Stack: (B, n_exits)
        confs_stack = torch.stack(confs_per_exit, dim=1)
        # Optionally sharpen/flatten confidences
        confs_pow = confs_stack ** self.p_conf

        # Predicted exit index = argmax over exits of confidence
        exit_pred = confs_pow.argmax(dim=1)  # (B,)

        # L1 distance between target and predicted indices
        exit_targets = exit_targets.view_as(exit_pred)
        loss = (exit_targets.to(exit_pred.dtype) - exit_pred).abs().float().mean()
        return loss
