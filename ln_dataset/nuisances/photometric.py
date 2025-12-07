from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

from .base import LocalNuisance
from ..config import get_severity_levels, SeverityLevel
from ..core.severity import sample_num_stickers, sample_areas
from ..core.geometry import sample_sticker_positions


def _sample_rects(
    image: Image.Image,
    level: SeverityLevel,
    rng: np.random.Generator,
):
    w, h = image.size
    img_area = w * h
    n = sample_num_stickers(level, rng)
    areas = sample_areas(level, n, img_area, rng)
    aspect_ratios = [1.0] * n
    rects, area_total, center_cov = sample_sticker_positions(
        level=level,
        areas=areas,
        image_size=(w, h),
        aspect_ratios=aspect_ratios,
        rng=rng,
    )
    return rects, areas, area_total, center_cov, w, h
