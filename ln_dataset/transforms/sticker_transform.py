from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F

from ..nuisances.sticker import StickerNuisance
from ..config import get_severity_levels


class StickerLN(torch.nn.Module):
    """
    Torchvision-style transform that randomly applies sticker nuisance.

    Usage:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            StickerLN(assets_root=".../assets/stickers", max_severity=3, p=0.5),
            transforms.ToTensor(),
        ])
    """

    def __init__(
        self,
        assets_root: str,
        max_severity: int = 3,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__()
        levels = get_severity_levels(list(range(1, max_severity + 1)))
        self.nuisance = StickerNuisance(assets_dir=Path(assets_root), schedule=levels)
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def forward(self, img):
        if torch.is_tensor(img):
            # Convert tensor [C,H,W] to PIL
            img = F.to_pil_image(img)

        if self.rng.random() > self.p:
            return img

        severity = int(self.rng.integers(1, len(self.nuisance.schedule) + 1))
        out, _ = self.nuisance.apply(img, severity=severity, rng=self.rng)
        return out
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F

from ..nuisances.sticker import StickerNuisance
from ..config import get_severity_levels


class StickerLN(torch.nn.Module):
    """
    Torchvision-style transform that randomly applies sticker nuisance.

    Usage:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            StickerLN(assets_root=".../assets/stickers", max_severity=3, p=0.5),
            transforms.ToTensor(),
        ])
    """

    def __init__(
        self,
        assets_root: str,
        max_severity: int = 3,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__()
        levels = get_severity_levels(list(range(1, max_severity + 1)))
        self.nuisance = StickerNuisance(assets_dir=Path(assets_root), schedule=levels)
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def forward(self, img):
        if torch.is_tensor(img):
            # Convert tensor [C,H,W] to PIL
            img = F.to_pil_image(img)

        if self.rng.random() > self.p:
            return img

        severity = int(self.rng.integers(1, len(self.nuisance.schedule) + 1))
        out, _ = self.nuisance.apply(img, severity=severity, rng=self.rng)
        return out
