"""The recogniser's batches: PaddleOCR's own recipe.

``tools/infer/predict_rec.py`` ``TextRecognizer.__call__`` (PaddleOCR) sorts the crops by width
over height, takes ``rec_batch_num`` at a time and pads each batch to its widest crop (at least
the model's 320 pixels). Measured against the one-crop-per-call path of the Triton pipeline on three
public scene-text sets (``docs/OCR.md``) it reads better,
because a short word gets more context than the 48-pixel minimum width. A crop is resized to
``min(ceil(48 * ratio), batch width)``; one wider than the engine's profile is squeezed, as the
reference does past its own limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence


REC_HEIGHT = 48
MIN_BATCH_WIDTH = 320  # rec_image_shape [3, 48, 320]


@dataclass(frozen=True)
class Batch:
    indices: list[int]  # into the caller's crops, narrowest first
    width: int  # every crop of the batch is padded to this


def plan_batches(ratios: Sequence[float], batch: int, *, max_width: int) -> list[Batch]:
    """Batches of at most ``batch`` crops by ascending width over height; each batch's width is
    ``int(48 * max(320 / 48, widest ratio))`` capped at ``max_width``."""
    order = np.argsort(np.asarray(ratios, np.float64), kind='stable')
    out = []
    for start in range(0, len(ratios), batch):
        idx = [int(i) for i in order[start : start + batch]]
        widest = max(MIN_BATCH_WIDTH / REC_HEIGHT, *(float(ratios[i]) for i in idx))
        out.append(Batch(idx, min(int(REC_HEIGHT * widest), max_width)))
    return out


def resized_width(ratio: float, batch_width: int) -> int:
    """The width a crop of this width over height is resized to inside a batch."""
    return min(math.ceil(REC_HEIGHT * ratio), batch_width)
