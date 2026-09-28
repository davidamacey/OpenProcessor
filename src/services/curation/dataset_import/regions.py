"""Region-class label attachment (W10.7): a dataset's region-class boxes
(e.g. "wheel") attach to their parent item (e.g. "car") by containment.

Pure — no I/O. ``dataset_import/job.py`` calls this per image, then
writes the result through W8's ``boxes_write_fields``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from src.services.curation.dataset_import.scan import LabelBox


BBox = tuple[float, float, float, float]


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: BBox, b: BBox) -> BBox | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def containment(box: BBox, parent: BBox) -> float:
    """``area(box ∩ parent) / area(box)``."""
    box_area = _area(box)
    if box_area <= 0:
        return 0.0
    inter = _intersection(box, parent)
    if inter is None:
        return 0.0
    return _area(inter) / box_area


def iou(a: BBox, b: BBox) -> float:
    inter = _intersection(a, b)
    if inter is None:
        return 0.0
    inter_area = _area(inter)
    union = _area(a) + _area(b) - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass(frozen=True)
class ParentCandidate:
    key: str
    """Caller-assigned stable id (e.g. the parent item's crop_id)."""
    bbox_norm: BBox
    class_name: str | None = None


@dataclass
class Attachment:
    parent_key: str
    box: LabelBox


@dataclass
class AttachResult:
    attachments: list[Attachment] = field(default_factory=list)
    """Region boxes matched to a parent."""
    standalone: list[LabelBox] = field(default_factory=list)
    """Region boxes with no qualifying parent — become standalone items."""


def parents_mode(
    mode: Literal['auto', 'labels', 'detect'], *, has_item_parent_labels: bool
) -> Literal['labels', 'detect']:
    """Resolve ``options.parents: "auto"`` (W10.7)."""
    if mode != 'auto':
        return mode
    return 'labels' if has_item_parent_labels else 'detect'


def attach_region_boxes(
    parents: list[ParentCandidate],
    boxes: list[LabelBox],
    *,
    containment_threshold: float = 0.9,
    parent_classes: frozenset[str] | None = None,
) -> AttachResult:
    """Attach each region-class box to the parent with the highest
    containment >= ``containment_threshold``. Ties: smaller parent area,
    then higher IoU, then lower ``key`` (deterministic). A box with no
    qualifying parent becomes standalone."""
    eligible = (
        parents
        if not parent_classes
        else [p for p in parents if p.class_name is None or p.class_name in parent_classes]
    )
    result = AttachResult()
    for box in boxes:
        best: ParentCandidate | None = None
        best_score: tuple[float, float, float, str] | None = None
        for parent in eligible:
            score = containment(box.bbox_norm, parent.bbox_norm)
            if score < containment_threshold:
                continue
            key = (
                -score,
                _area(parent.bbox_norm),
                -iou(box.bbox_norm, parent.bbox_norm),
                parent.key,
            )
            if best_score is None or key < best_score:
                best_score = key
                best = parent
        if best is None:
            result.standalone.append(box)
        else:
            result.attachments.append(Attachment(parent_key=best.key, box=box))
    return result


__all__ = [
    'AttachResult',
    'Attachment',
    'ParentCandidate',
    'attach_region_boxes',
    'containment',
    'iou',
    'parents_mode',
]
