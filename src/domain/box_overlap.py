"""Overlap of two boxes: the one IoU the claim matching uses (15a section 3.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final


if TYPE_CHECKING:
    from collections.abc import Sequence

LABEL_IOU_MATCH: Final = 0.5
"""A claim finds its subject again in a new detection at this overlap or better."""


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes in one coordinate frame."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0
