from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import yaml
from PIL import Image

from ..config import get_severity_levels, SeverityLevel
from ..core.severity import sample_num_stickers, sample_areas
from ..core.geometry import sample_sticker_positions, Rect
from ..core.compositing import alpha_blend_rgba


@dataclass
class StickerAsset:
    name: str
    path: Path
    aspect_ratio: float  # width / height


class StickerNuisance:
    """
    Sticker-based local nuisance using fixed sticker assets.
    """

    def __init__(
        self,
        assets_dir: Path,
        schedule: Dict[int, SeverityLevel] = None,
    ):
        """
        Args:
            assets_dir: directory containing stickers.yaml and PNG assets.
        """
        self.assets_dir = Path(assets_dir)
        self.schedule = get_severity_levels() if schedule is None else schedule
        self.assets = self._load_assets(self.assets_dir)

    @staticmethod
    def _load_assets(assets_dir: Path) -> List[StickerAsset]:
        cfg_path = assets_dir / "stickers.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing stickers.yaml in {assets_dir}")

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assets: List[StickerAsset] = []
        for item in cfg.get("stickers", []):
            name = item["name"]
            filename = item["filename"]
            path = assets_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as im:
                w, h = im.size
            ar = float(w) / float(h)
            assets.append(StickerAsset(name=name, path=path, aspect_ratio=ar))
        if not assets:
            raise ValueError("No sticker assets defined in stickers.yaml")
        return assets

    def _sample_assets(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> List[StickerAsset]:
        idxs = rng.integers(0, len(self.assets), size=n)
        return [self.assets[i] for i in idxs]

    def apply(
        self,
        image: Image.Image,
        severity: int,
        rng: np.random.Generator,
    ):
        if severity not in self.schedule:
            raise ValueError(f"Unsupported severity {severity}")

        level = self.schedule[severity]
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        img_area = w * h

        # Sample number of stickers and stencil areas
        n = sample_num_stickers(level, rng)
        areas = sample_areas(level, n, img_area, rng)
        chosen_assets = self._sample_assets(n, rng)
        aspect_ratios = [a.aspect_ratio for a in chosen_assets]

        # Sample positions consistent with level constraints
        rects, area_total, center_cov = sample_sticker_positions(
            level=level,
            areas=areas,
            image_size=(w, h),
            aspect_ratios=aspect_ratios,
            rng=rng,
        )

        # Composite stickers onto image
        out = image.copy()
        sticker_metadata: List[Dict[str, Any]] = []

        for asset, rect, area_frac in zip(chosen_assets, rects, areas):
            with Image.open(asset.path) as sticker:
                # overlay with alpha blending
                out = alpha_blend_rgba(out, sticker, rect)

            sticker_metadata.append(
                {
                    "name": asset.name,
                    "rel_path": str(asset.path.name),
                    "x0": rect.x0 / w,
                    "y0": rect.y0 / h,
                    "x1": rect.x1 / w,
                    "y1": rect.y1 / h,
                    "area_frac": area_frac,
                }
            )

        meta = {
            "n_stickers": len(chosen_assets),
            "severity": severity,
            "area_total": area_total,
            "center_coverage": center_cov,
            "stickers": sticker_metadata,
        }

        return out, meta
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import yaml
from PIL import Image

from ..config import get_severity_levels, SeverityLevel
from ..core.severity import sample_num_stickers, sample_areas
from ..core.geometry import sample_sticker_positions, Rect
from ..core.compositing import alpha_blend_rgba


@dataclass
class StickerAsset:
    name: str
    path: Path
    aspect_ratio: float  # width / height


class StickerNuisance:
    """
    Sticker-based local nuisance using fixed sticker assets.
    """

    def __init__(
        self,
        assets_dir: Path,
        schedule: Dict[int, SeverityLevel] = None,
    ):
        """
        Args:
            assets_dir: directory containing stickers.yaml and PNG assets.
        """
        self.assets_dir = Path(assets_dir)
        self.schedule = get_severity_levels() if schedule is None else schedule
        self.assets = self._load_assets(self.assets_dir)

    @staticmethod
    def _load_assets(assets_dir: Path) -> List[StickerAsset]:
        cfg_path = assets_dir / "stickers.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing stickers.yaml in {assets_dir}")

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assets: List[StickerAsset] = []
        for item in cfg.get("stickers", []):
            name = item["name"]
            filename = item["filename"]
            path = assets_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as im:
                w, h = im.size
            ar = float(w) / float(h)
            assets.append(StickerAsset(name=name, path=path, aspect_ratio=ar))
        if not assets:
            raise ValueError("No sticker assets defined in stickers.yaml")
        return assets

    def _sample_assets(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> List[StickerAsset]:
        idxs = rng.integers(0, len(self.assets), size=n)
        return [self.assets[i] for i in idxs]

    def apply(
        self,
        image: Image.Image,
        severity: int,
        rng: np.random.Generator,
    ):
        if severity not in self.schedule:
            raise ValueError(f"Unsupported severity {severity}")

        level = self.schedule[severity]
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        img_area = w * h

        # Sample number of stickers and stencil areas
        n = sample_num_stickers(level, rng)
        areas = sample_areas(level, n, img_area, rng)
        chosen_assets = self._sample_assets(n, rng)
        aspect_ratios = [a.aspect_ratio for a in chosen_assets]

        # Sample positions consistent with level constraints
        rects, area_total, center_cov = sample_sticker_positions(
            level=level,
            areas=areas,
            image_size=(w, h),
            aspect_ratios=aspect_ratios,
            rng=rng,
        )

        # Composite stickers onto image
        out = image.copy()
        sticker_metadata: List[Dict[str, Any]] = []

        for asset, rect, area_frac in zip(chosen_assets, rects, areas):
            with Image.open(asset.path) as sticker:
                # overlay with alpha blending
                out = alpha_blend_rgba(out, sticker, rect)

            sticker_metadata.append(
                {
                    "name": asset.name,
                    "rel_path": str(asset.path.name),
                    "x0": rect.x0 / w,
                    "y0": rect.y0 / h,
                    "x1": rect.x1 / w,
                    "y1": rect.y1 / h,
                    "area_frac": area_frac,
                }
            )

        meta = {
            "n_stickers": len(chosen_assets),
            "severity": severity,
            "area_total": area_total,
            "center_coverage": center_cov,
            "stickers": sticker_metadata,
        }

        return out, meta
