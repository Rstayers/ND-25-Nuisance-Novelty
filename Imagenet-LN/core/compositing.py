from typing import Tuple

import numpy as np
from PIL import Image

from .geometry import Rect


def alpha_blend_rgba(
    base: Image.Image,
    overlay: Image.Image,
    rect: Rect,
) -> Image.Image:
    """
    Alpha-blend RGBA overlay into base at the given Rect (pixel coords).
    Rect defines where the overlay should land in base coordinates.
    """
    if base.mode != "RGB":
        base = base.convert("RGB")

    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")

    base_np = np.array(base)
    overlay_np = np.array(overlay)

    x0, y0 = int(rect.x0), int(rect.y0)
    x1, y1 = int(rect.x1), int(rect.y1)

    # Sanity: resize overlay to match rect size
    target_w = max(1, x1 - x0)
    target_h = max(1, y1 - y0)
    if overlay_np.shape[1] != target_w or overlay_np.shape[0] != target_h:
        overlay = overlay.resize((target_w, target_h), Image.BILINEAR)
        overlay_np = np.array(overlay)

    # Extract alpha, normalize to [0, 1]
    alpha = overlay_np[..., 3:4] / 255.0
    rgb_overlay = overlay_np[..., :3]

    roi = base_np[y0:y1, x0:x1, :]
    blended = (alpha * rgb_overlay + (1.0 - alpha) * roi).astype("uint8")
    base_np[y0:y1, x0:x1, :] = blended

    return Image.fromarray(base_np, mode="RGB")
from typing import Tuple

import numpy as np
from PIL import Image

from .geometry import Rect


def alpha_blend_rgba(
    base: Image.Image,
    overlay: Image.Image,
    rect: Rect,
) -> Image.Image:
    """
    Alpha-blend RGBA overlay into base at the given Rect (pixel coords).
    Rect defines where the overlay should land in base coordinates.
    """
    if base.mode != "RGB":
        base = base.convert("RGB")

    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")

    base_np = np.array(base)
    overlay_np = np.array(overlay)

    x0, y0 = int(rect.x0), int(rect.y0)
    x1, y1 = int(rect.x1), int(rect.y1)

    # Sanity: resize overlay to match rect size
    target_w = max(1, x1 - x0)
    target_h = max(1, y1 - y0)
    if overlay_np.shape[1] != target_w or overlay_np.shape[0] != target_h:
        overlay = overlay.resize((target_w, target_h), Image.BILINEAR)
        overlay_np = np.array(overlay)

    # Extract alpha, normalize to [0, 1]
    alpha = overlay_np[..., 3:4] / 255.0
    rgb_overlay = overlay_np[..., :3]

    roi = base_np[y0:y1, x0:x1, :]
    blended = (alpha * rgb_overlay + (1.0 - alpha) * roi).astype("uint8")
    base_np[y0:y1, x0:x1, :] = blended

    return Image.fromarray(base_np, mode="RGB")
