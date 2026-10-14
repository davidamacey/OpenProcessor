"""Box geometry for the region evaluator: IoU, YOLO label parsing, matching.

Pure functions over normalized ``(x1, y1, x2, y2)`` boxes, plus the
stored-frame -> source-frame mapping (``to_source_frame``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


Box = tuple[float, float, float, float]

SOURCE_FRAMES: frozenset[str] = frozenset({'source'})
CROP_FRAMES: frozenset[str] = frozenset({'crop', 'item'})


class RegionFrameError(ValueError):
    """A region box is stored in a coordinate frame this evaluator cannot map."""


def as_box(value: Any) -> Box | None:
    """``[x1, y1, x2, y2]`` -> tuple, or None when absent / malformed / degenerate."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> Box:
    """YOLO ``cx, cy, w, h`` (normalized) -> ``x1, y1, x2, y2``, clamped to 0..1."""

    def _c(v: float) -> float:
        return min(max(v, 0.0), 1.0)

    return (_c(cx - w / 2), _c(cy - h / 2), _c(cx + w / 2), _c(cy + h / 2))


def parse_yolo_labels(text: str, class_ids: Iterable[int] | None = None) -> list[Box]:
    """Parse a YOLO label file into normalized xyxy boxes.

    Detection rows (``cls cx cy w h``) and segmentation rows (``cls x1 y1 x2
    y2 ...``, reduced to their bounding box) are both accepted. ``class_ids``
    keeps only those classes (``None`` = every row).
    """
    wanted = None if class_ids is None else {int(c) for c in class_ids}
    out: list[Box] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith('#'):
            continue
        try:
            cls = int(float(parts[0]))
            nums = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if wanted is not None and cls not in wanted:
            continue
        if len(nums) == 4:
            box = yolo_to_xyxy(*nums)
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            xs, ys = nums[0::2], nums[1::2]
            box = (min(xs), min(ys), max(xs), max(ys))
        else:
            continue
        if box[2] > box[0] and box[3] > box[1]:
            out.append(box)
    return out


def greedy_match(
    gt: Sequence[Box], pred: Sequence[Box], thr: float
) -> list[tuple[int, int, float]]:
    """One-to-one matching: highest-IoU pairs first, each box used once, IoU >= ``thr``."""
    pairs = sorted(
        (
            (v, gi, pi)
            for gi, g in enumerate(gt)
            for pi, p in enumerate(pred)
            if (v := iou(g, p)) >= thr and v > 0.0
        ),
        reverse=True,
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    out = []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, v))
    return out


def to_source_frame(box: Box, frame: str | None, item_box: Box | None) -> Box:
    """Map a stored region box to source-frame coordinates, or refuse."""
    if frame is None or frame in SOURCE_FRAMES:
        return box
    if frame in CROP_FRAMES:
        if item_box is None:
            raise RegionFrameError(
                f'region box is in frame {frame!r} but its item has no bbox_norm to map it through'
            )
        return crop_norm_to_source_norm(box, item_box)
    raise RegionFrameError(
        f'unknown region bbox frame {frame!r} (known: '
        f'{sorted(SOURCE_FRAMES | CROP_FRAMES)}); refusing to score boxes in an unknown '
        'coordinate system'
    )
