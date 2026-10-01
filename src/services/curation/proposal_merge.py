"""Merge machine proposals into an image's existing items (W10.6).

One pure function shared by the two callers that run a detector over an
image that may already hold labeled items:

* ``processing: "propose"`` on a dataset import (the detector looks for
  what the dataset's labels missed);
* the ``detect`` reprocess scope (re-run the detector on an indexed image).

The rule both enforce: **a proposal never overwrites a locked (human or
imported) label or box.** A proposal that overlaps a locked item is merged
into it (a ``proposal_chain`` note, plus a ``class_mismatch`` record when the
classes differ); one that overlaps an unlocked machine item refreshes it;
an unmatched one becomes a new machine item.

No I/O, no OpenSearch: the caller supplies :class:`ExistingItem` rows built
from the image's docs (``locked`` comes from
:func:`~src.clients.occ.is_locked_item`) and applies the returned plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.services.detection.geometry import iou


BBox = tuple[float, float, float, float]

LABEL_IOU_MATCH = 0.5

# ``kind`` values of the model-vs-label disagreement records.
#   class_mismatch       -- a label and a proposal overlap (IoU >= LABEL_IOU_MATCH)
#                           but the detector said a different class.
#   missed_label         -- a locked label no proposal overlaps: the detector missed it.
#   unmatched_detection  -- a proposal no label overlaps (on a reviewed-negative
#                           frame, every detection is one).
DISAGREEMENT_CLASS_MISMATCH = 'class_mismatch'
DISAGREEMENT_MISSED_LABEL = 'missed_label'
DISAGREEMENT_UNMATCHED_DETECTION = 'unmatched_detection'

# Same-box tolerance (normalized source frame): an unlocked item whose box
# a fresh proposal reproduces to well under a pixel is refreshed in place
# (same crop_id); a moved box is a replacement.
SAME_BOX_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ExistingItem:
    crop_id: str
    bbox_norm: BBox
    class_name: str | None
    locked: bool
    """The item carries any locked part (class, box, region verdict)."""


@dataclass(frozen=True)
class Proposal:
    bbox_norm: BBox
    class_id: int | None
    class_name: str | None
    score: float


@dataclass(frozen=True)
class Disagreement:
    kind: str
    crop_id: str | None
    bbox_norm: BBox
    label_class: str | None = None
    detector_class: str | None = None
    iou: float | None = None


@dataclass
class MergePlan:
    merged_into_locked: list[tuple[str, Proposal]] = field(default_factory=list)
    """``(locked crop_id, proposal)``: note the match on the locked item."""
    refreshed: list[tuple[str, Proposal]] = field(default_factory=list)
    """Unlocked item, same box: refresh in place."""
    replaced: list[tuple[str, Proposal]] = field(default_factory=list)
    """Unlocked item, moved box: delete the old item, create the proposal."""
    created: list[Proposal] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    """Unlocked items nothing matched (``remove_stale`` only)."""
    disagreements: list[Disagreement] = field(default_factory=list)


def count_disagreements(records: list[Disagreement]) -> dict[str, int]:
    """``{mismatches, missed_labels, unmatched_detections}`` for a record list."""
    kinds = [r.kind for r in records]
    return {
        'mismatches': kinds.count(DISAGREEMENT_CLASS_MISMATCH),
        'missed_labels': kinds.count(DISAGREEMENT_MISSED_LABEL),
        'unmatched_detections': kinds.count(DISAGREEMENT_UNMATCHED_DETECTION),
    }


def _same_box(a: BBox, b: BBox) -> bool:
    return all(abs(x - y) <= SAME_BOX_TOLERANCE for x, y in zip(a, b, strict=True))


def merge_item_proposals(
    existing: list[ExistingItem],
    proposals: list[Proposal],
    *,
    remove_stale: bool = False,
    on_negative_frame: bool = False,
) -> MergePlan:
    """Match ``proposals`` to ``existing`` items and return the plan.

    Matching is greedy by IoU, highest first, ties by (``crop_id``, proposal
    order): each existing item takes at most one proposal and each proposal
    at most one item, so the result is deterministic. ``remove_stale``
    (the ``detect`` reprocess scope) also lists unlocked items no proposal
    matched; a locked item is never listed, only reported as a
    ``missed_label``. ``on_negative_frame`` reports an unmatched proposal as
    an ``unmatched_detection`` (a likely false positive).
    """
    pairs: list[tuple[float, str, int, int]] = []
    for ei, item in enumerate(existing):
        for pi, prop in enumerate(proposals):
            value = iou(item.bbox_norm, prop.bbox_norm)
            if value >= LABEL_IOU_MATCH:
                pairs.append((-value, item.crop_id, pi, ei))
    pairs.sort()

    plan = MergePlan()
    taken_items: set[int] = set()
    taken_props: set[int] = set()
    for neg_iou, _cid, pi, ei in pairs:
        if ei in taken_items or pi in taken_props:
            continue
        taken_items.add(ei)
        taken_props.add(pi)
        item, prop = existing[ei], proposals[pi]
        if item.locked:
            plan.merged_into_locked.append((item.crop_id, prop))
            if item.class_name != prop.class_name:
                plan.disagreements.append(
                    Disagreement(
                        DISAGREEMENT_CLASS_MISMATCH,
                        item.crop_id,
                        item.bbox_norm,
                        label_class=item.class_name,
                        detector_class=prop.class_name,
                        iou=-neg_iou,
                    )
                )
        elif _same_box(item.bbox_norm, prop.bbox_norm):
            plan.refreshed.append((item.crop_id, prop))
        else:
            plan.replaced.append((item.crop_id, prop))

    for pi, prop in enumerate(proposals):
        if pi in taken_props:
            continue
        plan.created.append(prop)
        if on_negative_frame or any(item.locked for item in existing):
            plan.disagreements.append(
                Disagreement(
                    DISAGREEMENT_UNMATCHED_DETECTION,
                    None,
                    prop.bbox_norm,
                    detector_class=prop.class_name,
                )
            )

    for ei, item in enumerate(existing):
        if ei in taken_items:
            continue
        if item.locked:
            plan.disagreements.append(
                Disagreement(
                    DISAGREEMENT_MISSED_LABEL,
                    item.crop_id,
                    item.bbox_norm,
                    label_class=item.class_name,
                )
            )
        elif remove_stale:
            plan.removed.append(item.crop_id)
    return plan


__all__ = [
    'DISAGREEMENT_CLASS_MISMATCH',
    'DISAGREEMENT_MISSED_LABEL',
    'DISAGREEMENT_UNMATCHED_DETECTION',
    'LABEL_IOU_MATCH',
    'Disagreement',
    'ExistingItem',
    'MergePlan',
    'Proposal',
    'count_disagreements',
    'merge_item_proposals',
]
