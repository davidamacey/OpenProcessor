"""Image preprocessing shared by the cascade detectors."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from PIL import Image, ImageOps, UnidentifiedImageError

from src.services.detection.geometry import letterbox_to_square


if TYPE_CHECKING:
    import numpy as np


# =============================================================================
# Preprocessing helpers
# =============================================================================


def _decode_jpeg(jpeg_bytes: bytes) -> Image.Image:
    """Decode JPEG bytes to an EXIF-transposed RGB PIL image.

    Raises:
        ValueError: If the bytes cannot be parsed as an image.
    """
    if not jpeg_bytes:
        msg = 'empty crop bytes'
        raise ValueError(msg)
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except UnidentifiedImageError as exc:
        msg = f'crop is not a recognizable image: {exc}'
        raise ValueError(msg) from exc
    return img


def _letterbox(
    img: Image.Image,
    target: int = 640,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox a PIL image to ``target`` by ``target`` for the detector.

    Thin wrapper over :func:`src.services.detection.geometry.letterbox_to_square`
    (shared with the curation ingest service) kept here so call sites in
    this module don't need to change; see that function for the return
    shape contract.
    """
    return letterbox_to_square(img, target=target, fill=fill)
