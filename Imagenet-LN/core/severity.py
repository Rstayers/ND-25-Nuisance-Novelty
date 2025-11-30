from typing import List
import numpy as np

from ..config import SeverityLevel


def sample_num_stickers(level: SeverityLevel, rng: np.random.Generator) -> int:
    return int(rng.integers(level.n_min, level.n_max + 1))


def sample_areas(
    level: SeverityLevel,
    n_stickers: int,
    image_area: int,
    rng: np.random.Generator,
) -> List[float]:
    """
    Sample per-sticker area fractions (relative to full image area) that
    sum into [level.area_min, level.area_max].
    """
    # Start with Dirichlet for relative weights.
    weights = rng.dirichlet([1.0] * n_stickers)
    total_area = rng.uniform(level.area_min, level.area_max)
    areas = total_area * weights
    # Clip in case of numerical issues.
    areas = np.clip(areas, 1e-4, None)
    return areas.tolist()
from typing import List
import numpy as np

from ..config import SeverityLevel


def sample_num_stickers(level: SeverityLevel, rng: np.random.Generator) -> int:
    return int(rng.integers(level.n_min, level.n_max + 1))


def sample_areas(
    level: SeverityLevel,
    n_stickers: int,
    image_area: int,
    rng: np.random.Generator,
) -> List[float]:
    """
    Sample per-sticker area fractions (relative to full image area) that
    sum into [level.area_min, level.area_max].
    """
    # Start with Dirichlet for relative weights.
    weights = rng.dirichlet([1.0] * n_stickers)
    total_area = rng.uniform(level.area_min, level.area_max)
    areas = total_area * weights
    # Clip in case of numerical issues.
    areas = np.clip(areas, 1e-4, None)
    return areas.tolist()
