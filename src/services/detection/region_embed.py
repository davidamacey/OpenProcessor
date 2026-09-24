"""PE-Core embedding for a region-of-interest crop (LG-1).

The legacy stack ran a standalone backfill loop that cropped the region,
encoded it, L2-normalized, and wrote the vector back — `region_embedding`
is what region-FP clustering (:mod:`src.services.curation.clustering.orchestrator`,
region-FP paths) and ``/regions/suspected_false_positives``
(:mod:`src.routers.curation.regions_fp`) both key off. Nothing in ``main``
writes it yet.

This module owns just the "crop -> vector" step so it can be shared
between a live writer (the detection worker, at verify time) and the
one-off backfill script for existing items — see
``scripts/curation/backfill_region_embeddings.py``.

CM-3 must land before this is used for anything that feeds
``FalsePositiveCentroidStore`` — the store's search() previously
returned squared L2, which every consumer assumed was plain L2.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.core.logging import get_logger


if TYPE_CHECKING:
    from src.clients.pe_encoder import PEEncoder

logger = get_logger(__name__)


def decode_region_jpeg(region_jpeg: bytes) -> np.ndarray | None:
    """JPEG bytes -> HWC uint8 RGB array, or ``None`` on a decode failure.

    A decode failure is treated as non-fatal by every caller here — a
    missing region embedding just means that item is skipped this pass
    (the live writer) or left for the next backfill run (the script);
    neither should crash on a single corrupt crop.
    """
    try:
        img = Image.open(io.BytesIO(region_jpeg))
        img.load()
        rgb = img if img.mode == 'RGB' else img.convert('RGB')
        return np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning('region_embed_decode_failed', error=str(exc))
        return None


async def embed_region_crops(
    pe: PEEncoder | Any, region_jpegs: list[bytes]
) -> list[list[float] | None]:
    """PE-Core embeddings for a batch of region-crop JPEGs.

    Returns one entry per input, in order: a 1024-float L2-normalized
    list (``PEEncoder.embed_crops`` already normalizes), or ``None`` for
    an entry that failed to decode. Encoding failures (a Triton call
    that raises) propagate — a systemic encoder failure should stop the
    caller's batch rather than silently writing nothing for every item.
    """
    decoded: list[np.ndarray | None] = [decode_region_jpeg(j) for j in region_jpegs]
    good_indices = [i for i, arr in enumerate(decoded) if arr is not None]
    if not good_indices:
        return [None] * len(region_jpegs)

    crops: list[np.ndarray] = [arr for arr in decoded if arr is not None]
    embeddings = await pe.embed_crops(crops)

    out: list[list[float] | None] = [None] * len(region_jpegs)
    for out_pos, i in enumerate(good_indices):
        out[i] = embeddings[out_pos].astype(float).tolist()
    return out


__all__ = ['decode_region_jpeg', 'embed_region_crops']
