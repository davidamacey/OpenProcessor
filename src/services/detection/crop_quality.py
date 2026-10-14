"""Primary-subject blur scoring for detected item crops.

Laplacian crop/full-image sharpness ratio, used at ingest and by backfill /
contact-sheet tooling to gate which crops surface for labeling and
(optionally) clustering.

Definitions (2-decimal rounding matches the reference calibration):

    full_var = round(Laplacian(gray_full, CV_64F).var(), 2)
    box_var  = round(Laplacian(gray_crop, CV_64F).var(), 2)
    ratio    = round(box_var / full_var, 2)
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from src.core.logging import get_logger


logger = get_logger(__name__)

# Below this Laplacian variance the source frame is effectively flat/blank;
# the ratio is undefined (divide-by-zero) so we report it as unknown.
_FULL_VAR_EPS = 1e-6


ColorOrder = Literal['bgr', 'rgb']
_GRAY_CODE = {'bgr': cv2.COLOR_BGR2GRAY, 'rgb': cv2.COLOR_RGB2GRAY}


def laplacian_var(gray: np.ndarray) -> float:
    """Return the variance of the Laplacian of a single-channel image.

    Args:
        gray: 2-D grayscale uint8 (or float) array.

    Returns:
        Variance of the Laplacian response, rounded to 2 decimals.

    The Laplacian of an 8-bit image is integer valued (at most 1,020 in magnitude), so
    for ``uint8`` input the variance comes from exact integer sums over a 16-bit
    response: a quarter of the memory of the float64 response, and the true variance
    where ``ndarray.var`` carries a rounding error near 1e-11. Both round to the same
    2 decimals, except for a value that lies within that error of a rounding tie.
    """
    if gray.dtype != np.uint8:
        return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    lap = cv2.Laplacian(gray, cv2.CV_16S)
    total = int(cv2.sumElems(lap)[0])
    squares = int(np.einsum('ij,ij->', lap, lap, dtype=np.int64))  # type: ignore[arg-type]  # numpy stubs type the int64 einsum overload as bool-only
    return variance_from_sums(lap.size, total, squares)


def variance_from_sums(n: int, total: int, squares: int) -> float:
    """The Laplacian variance from its exact integer sums (pixel count, sum, sum of squares),
    rounded as :func:`laplacian_var` rounds; the GPU decode reports these sums."""
    return round((n * squares - total * total) / (n * n), 2)


def image_lap_var(image: np.ndarray, color_order: ColorOrder = 'bgr') -> float:
    """Compute the full-image Laplacian variance (the ratio denominator).

    Args:
        image: 3-channel image array.
        color_order: Channel order of ``image``: ``'bgr'`` (OpenCV) or ``'rgb'`` (PIL, numpy).

    Returns:
        Full-image Laplacian variance, rounded to 2 decimals.
    """
    gray = cv2.cvtColor(image, _GRAY_CODE[color_order])
    return laplacian_var(gray)


def crop_lap_var(
    image: np.ndarray,
    bbox_px: tuple[float, float, float, float],
    color_order: ColorOrder = 'bgr',
) -> float | None:
    """Laplacian variance of a crop region, or ``None`` if degenerate.

    Clamps the bbox to image bounds and guards against zero-area crops.

    Args:
        image: Source image, 3 channels in ``color_order``.
        bbox_px: ``(x1, y1, x2, y2)`` pixel coordinates.
        color_order: Channel order of ``image``: ``'bgr'`` (OpenCV) or ``'rgb'`` (PIL, numpy).

    Returns:
        Crop Laplacian variance rounded to 2 decimals, or ``None``.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (round(v) for v in bbox_px)
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, _GRAY_CODE[color_order])
    return laplacian_var(gray)


def blur_ratio(box_var: float | None, full_var: float) -> float | None:
    """Crop/full Laplacian-variance ratio, or ``None`` when undefined.

    Args:
        box_var: Crop Laplacian variance (or ``None``).
        full_var: Full-image Laplacian variance.

    Returns:
        ``round(box_var / full_var, 2)`` or ``None`` if ``box_var`` is missing
        or ``full_var`` is ~0 (flat frame).
    """
    if box_var is None or full_var <= _FULL_VAR_EPS:
        return None
    return round(box_var / full_var, 2)
