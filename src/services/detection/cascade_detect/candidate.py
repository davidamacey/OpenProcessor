"""The :class:`RegionCandidate` result dataclass of the detection cascade."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionCandidate:
    """One sub-region detection in the crop's coordinate frame.

    Attributes:
        bbox_norm: ``(x1, y1, x2, y2)`` normalized to ``[0, 1]`` of the
            crop. Always axis-aligned with ``x2 > x1`` and ``y2 > y1``.
        score: Detector confidence in ``[0, 1]``.
        source: Detector identifier. Kept as a field so candidate
            objects from different detectors (SAM3, PaddleOCR) can be
            merged later without losing provenance.
        rectangularity: Mask-area / bbox-area ratio. ``None`` for a
            box-only detector; populated by mask-based detectors.
        mask_polygon: Normalized ``(x, y)`` outline in the same frame as
            ``bbox_norm``; only the full-image pass asks for it.
    """

    bbox_norm: tuple[float, float, float, float]
    score: float
    source: str = ''
    rectangularity: float | None = None
    mask_polygon: tuple[tuple[float, float], ...] | None = None
