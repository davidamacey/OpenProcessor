"""Sanity gate, class provenance and crop-to-source frame transform."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any


# =============================================================================
# Sanity gate + provenance helpers (Phase A2 / A3)
# =============================================================================


def _now_iso() -> str:
    """ISO-8601 UTC timestamp used for region_detected_at / class_labeled_at."""
    return datetime.now(UTC).isoformat()


def is_plausible_region_bbox(
    region_in_crop: tuple[float, float, float, float],
    parent_in_source: tuple[float, float, float, float] | None = None,
) -> tuple[bool, str]:
    """Return ``(ok, reason)``; ``reason='ok'`` on pass.

    GEOMETRY GUARD ONLY. This gate does not apply aspect-ratio or
    region-vs-parent size heuristics: shape bands built from a single
    domain's assumptions (e.g. "this region type is always wider than
    tall") silently reject legitimate detections from other domains
    (foreshortened / angled views, naturally-square sub-regions). This
    gate runs BEFORE the VLM verification step, so a rejected box never
    gets a chance to be confirmed — the shape/not-shape decision belongs
    to the verifier and the detector confidence floors, not to a shape
    prior baked into the cascade.

    The region bbox is in the **item crop** coordinate frame (normalized
    to ``[0, 1]``). ``parent_in_source`` is kept for signature stability
    (all cascade call sites pass it); it's only used for a degeneracy
    check.

    Rejects only:

    * Non-finite or non-numeric coordinates.
    * Degenerate region box (``x2 <= x1`` or ``y2 <= y1``).
    * Degenerate parent bbox, when supplied.
    """
    try:
        x1, y1, x2, y2 = region_in_crop
    except (TypeError, ValueError):
        return False, 'bbox_unpack_failed'
    for name, v in (('x1', x1), ('y1', y1), ('x2', x2), ('y2', y2)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False, f'{name}_not_numeric'
        if not math.isfinite(f):
            return False, f'{name}_non_finite'
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    if w <= 0.0 or h <= 0.0:
        return False, 'degenerate_zero_size'
    if parent_in_source is not None:
        try:
            vx1, vy1, vx2, vy2 = parent_in_source
            vw = float(vx2) - float(vx1)
            vh = float(vy2) - float(vy1)
        except (TypeError, ValueError):
            return False, 'parent_bbox_unpack_failed'
        if vw <= 0.0 or vh <= 0.0:
            return False, 'parent_bbox_degenerate'
    return True, 'ok'


def class_provenance(
    detector: str,
    detector_version: str,
    *,
    labeler: str,
    labeled_at: str | None = None,
) -> dict[str, Any]:
    """Build the class-provenance dict for crop class label writers.

    Not RegionFields-governed — ``class_*`` fields are the item's class
    label provenance, orthogonal to the region-of-interest sub-annotation.
    """
    return {
        'class_detector': detector,
        'class_detector_version': detector_version,
        'class_labeler': labeler,
        'class_labeled_at': labeled_at or _now_iso(),
    }


# =============================================================================
# Coordinate-frame transforms
# =============================================================================


def crop_norm_to_source_norm(
    region_in_crop: tuple[float, float, float, float],
    parent_in_source: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Re-project a region bbox from the item-crop frame to the source-image frame.

    Both inputs and the output are normalized to ``[0, 1]``. The region
    bbox is in the crop's coordinate frame (which is what
    :class:`RegionDetector.detect` returns); the parent bbox tells us
    where that crop sits inside the original image. We need the
    source-image frame for YOLO training labels — Ultralytics expects
    every label row in the source image's coordinate system, regardless
    of any cropping we did during ingest.

    Math: if the parent box in source is ``(vx1, vy1, vx2, vy2)`` and
    the region in crop is ``(px1, py1, px2, py2)``, then region in
    source is::

        region_x1 = vx1 + px1 * (vx2 - vx1)
        region_y1 = vy1 + py1 * (vy2 - vy1)
        region_x2 = vx1 + px2 * (vx2 - vx1)
        region_y2 = vy1 + py2 * (vy2 - vy1)

    The result is clamped to ``[0, 1]`` defensively.

    Args:
        region_in_crop: ``(x1, y1, x2, y2)`` of the region, normalized to
            the item crop's frame.
        parent_in_source: ``(x1, y1, x2, y2)`` of the parent box,
            normalized to the source image's frame.

    Returns:
        ``(x1, y1, x2, y2)`` of the region, normalized to the source image.
    """
    px1, py1, px2, py2 = region_in_crop
    vx1, vy1, vx2, vy2 = parent_in_source
    vw = vx2 - vx1
    vh = vy2 - vy1
    sx1 = max(0.0, min(1.0, vx1 + px1 * vw))
    sy1 = max(0.0, min(1.0, vy1 + py1 * vh))
    sx2 = max(0.0, min(1.0, vx1 + px2 * vw))
    sy2 = max(0.0, min(1.0, vy1 + py2 * vh))
    return (sx1, sy1, sx2, sy2)
