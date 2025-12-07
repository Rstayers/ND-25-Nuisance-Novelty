from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


@dataclass(frozen=True)
class SeverityLevel:
    severity: int
    n_min: int
    n_max: int
    area_min: float         # as fraction of image area
    area_max: float
    center_min: float       # fraction of central region covered
    center_max: float


# Normalized central region: [x0, y0, x1, y1] in [0, 1]
CENTRAL_REGION: Tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75)

# Sticker MVP schedule, independent of image resolution.
DEFAULT_STICKER_SCHEDULE: Dict[int, SeverityLevel] = {
    1: SeverityLevel(
        severity=1, n_min=1, n_max=1,
        area_min=0.03, area_max=0.06,
        center_min=0.0, center_max=0.05
    ),
    2: SeverityLevel(
        severity=2, n_min=1, n_max=2,
        area_min=0.05, area_max=0.10,
        center_min=0.0, center_max=0.10
    ),
    3: SeverityLevel(
        severity=3, n_min=2, n_max=3,
        area_min=0.08, area_max=0.15,
        center_min=0.0, center_max=0.20
    ),
    4: SeverityLevel(
        severity=4, n_min=2, n_max=3,
        area_min=0.12, area_max=0.22,
        center_min=0.20, center_max=0.35
    ),
    5: SeverityLevel(
        severity=5, n_min=3, n_max=4,
        area_min=0.18, area_max=0.30,
        center_min=0.35, center_max=0.50
    ),
}

MAX_IMAGE_COVERAGE: float = 0.30  # hard upper bound on sticker area
MAX_SAMPLING_ATTEMPTS: int = 20   # per (image, severity)


def get_severity_levels(levels: Optional[List[int]] = None) -> Dict[int, SeverityLevel]:
    if levels is None:
        return DEFAULT_STICKER_SCHEDULE
    return {k: DEFAULT_STICKER_SCHEDULE[k] for k in levels}
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


@dataclass(frozen=True)
class SeverityLevel:
    severity: int
    n_min: int
    n_max: int
    area_min: float         # as fraction of image area
    area_max: float
    center_min: float       # fraction of central region covered
    center_max: float


# Normalized central region: [x0, y0, x1, y1] in [0, 1]
CENTRAL_REGION: Tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75)

# Sticker MVP schedule, independent of image resolution.
DEFAULT_STICKER_SCHEDULE: Dict[int, SeverityLevel] = {
    1: SeverityLevel(
        severity=1, n_min=1, n_max=1,
        area_min=0.03, area_max=0.06,
        center_min=0.0, center_max=0.05
    ),
    2: SeverityLevel(
        severity=2, n_min=1, n_max=2,
        area_min=0.05, area_max=0.10,
        center_min=0.0, center_max=0.10
    ),
    3: SeverityLevel(
        severity=3, n_min=2, n_max=3,
        area_min=0.08, area_max=0.15,
        center_min=0.0, center_max=0.20
    ),
    4: SeverityLevel(
        severity=4, n_min=2, n_max=3,
        area_min=0.12, area_max=0.22,
        center_min=0.20, center_max=0.35
    ),
    5: SeverityLevel(
        severity=5, n_min=3, n_max=4,
        area_min=0.18, area_max=0.30,
        center_min=0.35, center_max=0.50
    ),
}

MAX_IMAGE_COVERAGE: float = 0.30  # hard upper bound on sticker area
MAX_SAMPLING_ATTEMPTS: int = 100  # or 200


def get_severity_levels(levels: Optional[List[int]] = None) -> Dict[int, SeverityLevel]:
    if levels is None:
        return DEFAULT_STICKER_SCHEDULE
    return {k: DEFAULT_STICKER_SCHEDULE[k] for k in levels}
