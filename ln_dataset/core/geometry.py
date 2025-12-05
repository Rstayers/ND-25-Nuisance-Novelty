from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..config import CENTRAL_REGION, SeverityLevel, MAX_IMAGE_COVERAGE


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def clamp(self, w: int, h: int) -> "Rect":
        x0 = max(0, min(self.x0, w))
        x1 = max(0, min(self.x1, w))
        y0 = max(0, min(self.y0, h))
        y1 = max(0, min(self.y1, h))
        return Rect(x0, y0, x1, y1)

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


def _center_region_pixels(w: int, h: int) -> Rect:
    cx0, cy0, cx1, cy1 = CENTRAL_REGION
    return Rect(
        x0=cx0 * w,
        y0=cy0 * h,
        x1=cx1 * w,
        y1=cy1 * h,
    )


def compute_intersection_area(a: Rect, b: Rect) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def compute_center_coverage(sticker_rects: List[Rect], w: int, h: int) -> float:
    center = _center_region_pixels(w, h)
    center_area = center.area()
    if center_area <= 0:
        return 0.0
    covered = 0.0
    # Approximate union via inclusion–exclusion is overkill; union over
    # few rects can be approximated with sampling or simple over-estimate.
    # For MVP, we sum intersections but clip at center_area.
    for r in sticker_rects:
        covered += compute_intersection_area(r, center)
    covered = min(covered, center_area)
    return covered / center_area


def sample_sticker_positions(
    level: SeverityLevel,
    areas: List[float],
    image_size: Tuple[int, int],
    aspect_ratios: List[float],
    rng: np.random.Generator,
    max_attempts: int = 20,
) -> Tuple[List[Rect], float, float]:
    """
    Sample sticker rectangles (pixel coordinates) satisfying severity constraints.

    Returns:
        rects: list of Rect (len == len(areas))
        area_total: total sticker area / image area
        center_coverage: fraction of center region covered
    Raises:
        RuntimeError if constraints cannot be satisfied.
    """
    w, h = image_size
    img_area = float(w * h)
    n = len(areas)
    assert len(aspect_ratios) == n

    center = _center_region_pixels(w, h)

    for attempt in range(max_attempts):
        rects: List[Rect] = []
        total_area_px = 0.0

        for a_frac, ar in zip(areas, aspect_ratios):
            area_px = a_frac * img_area
            # width * height = area_px, width / height = ar => solve:
            # width = sqrt(area_px * ar), height = area_px / width
            width = np.sqrt(area_px * ar)
            height = area_px / width

            # sample center position
            cx = rng.uniform(0 + width / 2, w - width / 2)
            cy = rng.uniform(0 + height / 2, h - height / 2)

            rect = Rect(
                x0=cx - width / 2,
                y0=cy - height / 2,
                x1=cx + width / 2,
                y1=cy + height / 2,
            ).clamp(w, h)

            rects.append(rect)
            total_area_px += rect.area()

        area_total_frac = total_area_px / img_area
        if area_total_frac > MAX_IMAGE_COVERAGE:
            continue

        center_cov = compute_center_coverage(rects, w, h)

        if (
            level.center_min <= center_cov <= level.center_max
            and level.area_min <= area_total_frac <= level.area_max
        ):
            return rects, area_total_frac, center_cov

    raise RuntimeError("Failed to sample sticker positions satisfying constraints.")
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..config import CENTRAL_REGION, SeverityLevel, MAX_IMAGE_COVERAGE


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def clamp(self, w: int, h: int) -> "Rect":
        x0 = max(0, min(self.x0, w))
        x1 = max(0, min(self.x1, w))
        y0 = max(0, min(self.y0, h))
        y1 = max(0, min(self.y1, h))
        return Rect(x0, y0, x1, y1)

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


def _center_region_pixels(w: int, h: int) -> Rect:
    cx0, cy0, cx1, cy1 = CENTRAL_REGION
    return Rect(
        x0=cx0 * w,
        y0=cy0 * h,
        x1=cx1 * w,
        y1=cy1 * h,
    )


def compute_intersection_area(a: Rect, b: Rect) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def compute_center_coverage(sticker_rects: List[Rect], w: int, h: int) -> float:
    center = _center_region_pixels(w, h)
    center_area = center.area()
    if center_area <= 0:
        return 0.0
    covered = 0.0
    # Approximate union via inclusion–exclusion is overkill; union over
    # few rects can be approximated with sampling or simple over-estimate.
    # For MVP, we sum intersections but clip at center_area.
    for r in sticker_rects:
        covered += compute_intersection_area(r, center)
    covered = min(covered, center_area)
    return covered / center_area


def sample_sticker_positions(
    level: SeverityLevel,
    areas: List[float],
    image_size: Tuple[int, int],
    aspect_ratios: List[float],
    rng: np.random.Generator,
    max_attempts: int = 20,
) -> Tuple[List[Rect], float, float]:
    """
    Sample sticker rectangles (pixel coordinates) satisfying severity constraints.

    Returns:
        rects: list of Rect (len == len(areas))
        area_total: total sticker area / image area
        center_coverage: fraction of center region covered
    Raises:
        RuntimeError if constraints cannot be satisfied.
    """
    w, h = image_size
    img_area = float(w * h)
    n = len(areas)
    assert len(aspect_ratios) == n

    center = _center_region_pixels(w, h)

    for attempt in range(max_attempts):
        rects: List[Rect] = []
        total_area_px = 0.0

        for a_frac, ar in zip(areas, aspect_ratios):
            area_px = a_frac * img_area
            # width * height = area_px, width / height = ar => solve:
            # width = sqrt(area_px * ar), height = area_px / width
            width = np.sqrt(area_px * ar)
            height = area_px / width

            # sample center position
            cx = rng.uniform(0 + width / 2, w - width / 2)
            cy = rng.uniform(0 + height / 2, h - height / 2)

            rect = Rect(
                x0=cx - width / 2,
                y0=cy - height / 2,
                x1=cx + width / 2,
                y1=cy + height / 2,
            ).clamp(w, h)

            rects.append(rect)
            total_area_px += rect.area()

        area_total_frac = total_area_px / img_area
        if area_total_frac > MAX_IMAGE_COVERAGE:
            continue

        center_cov = compute_center_coverage(rects, w, h)

        if (
            level.center_min <= center_cov <= level.center_max
            and level.area_min <= area_total_frac <= level.area_max
        ):
            return rects, area_total_frac, center_cov

    raise RuntimeError("Failed to sample sticker positions satisfying constraints.")
