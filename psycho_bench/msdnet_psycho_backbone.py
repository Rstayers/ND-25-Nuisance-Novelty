# psycho_bench/msdnet_psycho_backbone.py
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
from types import SimpleNamespace

from msdnet import  MSDNet


class MSDNetPsycho(nn.Module):
    """
    MSDNet wrapper that:
      - exposes per-exit logits during training (for psychophysical loss)
      - returns only final-exit logits during evaluation (for OpenOOD)
    """

    def __init__(
        self,
        n_exits: int = 5,
        data: str = "ImageNet",
        base: int = 4,
        step: int = 4,
        stepmode: str = "even",
        nChannels: int = 32,
        growthRate: int = 6,
        grFactor: str = "1-2-4",
        prune: str = "max",
        bnFactor: str = "1-2-4",
        bottleneck: bool = True,
        reduction: float = 0.5,
    ):
        super().__init__()

        # Parse growth/bottleneck factors, set nScales
        gr_list = list(map(int, grFactor.split("-")))
        bn_list = list(map(int, bnFactor.split("-")))
        assert len(gr_list) == len(bn_list)
        nScales = len(gr_list)

        # Build the args namespace EXACTLY with the fields MSDNet expects
        args = SimpleNamespace(
            data=data,
            nBlocks=n_exits,
            base=base,
            step=step,
            stepmode=stepmode,
            nChannels=nChannels,
            nScales=nScales,
            growthRate=growthRate,
            grFactor=gr_list,
            prune=prune,
            reduction=reduction,
            bnFactor=bn_list,
            bottleneck=bottleneck,
        )

        self.args = args
        self.core = MSDNet(args)
        self.n_exits = n_exits

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        If return_all=False:
            returns final-exit logits only.
        If return_all=True:
            returns list of per-exit logits.
        """
        # MSDNet.forward(x) returns a list of logits (one per exit)
        logits_list: List[torch.Tensor] = self.core(x)

        if not return_all:
            return logits_list[-1]
        else:
            return logits_list
