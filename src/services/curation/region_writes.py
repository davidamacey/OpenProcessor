"""The update documents every human region writer applies.

``PUT /crops/{id}/regions``, ``PUT /crops/batch_regions``, ``PATCH
/crops/{id}/regions/{box_id}``, ``POST /regions/batch_box_state``, ``PATCH
/crops/{id}/region_meta`` and ``POST /regions/batch_status`` all build
their writes here, over the per-item box list
(:mod:`src.services.curation.region_boxes`). The box transitions
themselves live in :mod:`src.services.curation.region_box_edits`; this
module turns a new box list into the item-level update document, so the
region invariants hold whichever writer a client picks:

- every box write goes through
  :func:`~src.services.curation.region_boxes.boxes_write_fields`, so the
  list, the counts, ``max_score`` and ``region_revision`` never disagree;
- the item status is re-derived from the list;
- the confirm status needs an accepted box (``RegionBoxWriteError``).

Each writer returns the post-write item built by :func:`post_write_item`,
so a client adopts the server's state instead of re-deriving it.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from src.config import get_region_fields
from src.config.region_rejection import REJECT_REASON_VERIFIER
from src.config.region_state import CONFIRM_STATUS, REGION_STATUS_INFO, RegionStatus
from src.services.curation.region_box_edits import boxes_with_status
from src.services.curation.region_boxes import (
    RegionBox,
    RegionBoxWriteError,
    boxes_write_fields,
    derive_status,
    read_boxes,
)
from src.services.curation.wire import serialize_item
from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm


if TYPE_CHECKING:
    from collections.abc import Sequence


def parent_to_source_bbox(region_in_parent: Sequence[float], parent_bbox_norm: Any) -> list[float]:
    """Project a box drawn in the item-crop frame into the source frame.

    ``parent_bbox_norm`` is the item's own stored ``bbox_norm`` (source
    frame). Raises :class:`RegionBoxWriteError` when the item has no usable
    box to project through.
    """
    if not isinstance(parent_bbox_norm, list | tuple) or len(parent_bbox_norm) != 4:
        msg = 'item has no bbox_norm to project a parent-frame region through'
        raise RegionBoxWriteError(msg)
    try:
        px1, py1, px2, py2 = (float(v) for v in parent_bbox_norm)
    except (TypeError, ValueError) as exc:
        msg = 'item bbox_norm is not numeric'
        raise RegionBoxWriteError(msg) from exc
    if px2 <= px1 or py2 <= py1:
        msg = 'item bbox_norm is degenerate'
        raise RegionBoxWriteError(msg)
    x1, y1, x2, y2 = (float(v) for v in region_in_parent)
    return list(crop_norm_to_source_norm((x1, y1, x2, y2), (px1, py1, px2, py2)))


def human_box_write(
    current: dict[str, Any],
    boxes: Sequence[RegionBox],
    *,
    label_source: str,
    now: str,
    human_name: str,
    human_version: str,
) -> dict[str, Any]:
    """The update document for a human edit that produced ``boxes`` -- the
    one builder behind every box route (``PUT regions``, ``PUT
    batch_regions``, ``PATCH regions/{box_id}``, ``POST
    batch_box_state``).

    Status is re-derived from the list (an empty list is the human "no
    region visible"); the human is stamped as the verifier. W8.7: the
    write counts as the item's validation (``validated``, ``set_complete``)
    only when no box is left ``proposed`` -- a partial review leaves both
    as stored, so the item stays reviewable.
    """
    F = get_region_fields()
    doc = dict(boxes_write_fields(boxes, current_src=current))
    status = derive_status(boxes, empty_status=RegionStatus.NO_REGION_VISIBLE)
    doc.update(
        {
            F.status: status.value,
            F.label_source: label_source,
            F.verified: status == CONFIRM_STATUS,
            F.verified_at: now,
            F.verifier: human_name,
            F.verifier_version: human_version,
            'updated_at': now,
        }
    )
    if not any(b.state == 'proposed' for b in boxes):
        doc[F.validated] = True
        doc[F.set_complete] = True
    return doc


def label_source_is_human(label_source: str | None) -> bool:
    """A region edit is a person's when its ``region_label_source`` says so
    (``human``, ``human_move``, ...); an auto-relabel job names another source."""
    return (label_source or '').lower().startswith('human')


def human_status_box_write(
    region_status: str,
    current: dict[str, Any],
    *,
    rejection_reason: str | None = None,
    label_source: str | None = 'human',
) -> dict[str, Any]:
    """The ``region_boxes``-based whole-set status write backing ``PATCH
    /crops/{id}/region_meta`` and ``POST /regions/batch_status``.

    Delegates the box-list transition to
    :func:`~src.services.curation.region_box_edits.boxes_with_status`
    (raises :class:`~src.services.curation.region_boxes.RegionBoxWriteError`
    for confirming with no box to confirm). ``rejection_reason`` is the
    reviewer's note on every box a ``verify_rejected`` write rejects.

    A whole-set CONFIRM is the one deliberate exception to "a whole-set
    write never overrides a per-box decision": it reopens only the boxes
    the *verifier* rejected (W8-cleanup M3) -- see ``boxes_with_status``.
    When nothing is reopenable it raises ``no_accepted_box`` (422).

    The item-level ``rejection_reason`` mirror is derived by
    ``boxes_write_fields`` (only when no accepted/false-positive box
    exists). A box-less status that wants a reason (``no_region_visible``)
    has no box to carry it, so the reason is stored on the item (M1(a)).
    """
    F = get_region_fields()
    status = RegionStatus(region_status)
    if rejection_reason is None and not label_source_is_human(label_source):
        # A whole-set reject by an automated source is a machine verdict: the
        # default human reason would take the human lock on every box.
        rejection_reason = REJECT_REASON_VERIFIER
    new_boxes = boxes_with_status(
        status.value, read_boxes(current, F), rejection_reason=rejection_reason
    )
    doc: dict[str, Any] = dict(boxes_write_fields(new_boxes, current_src=current))
    # `empty_status=status`: only NO_REGION_VISIBLE ever leaves `new_boxes`
    # empty (boxes_with_status returns `[]` for it) -- every other status
    # is reflected by derive_status's own precedence over the now-uniform
    # box list, so this only matters for that one case.
    doc[F.status] = derive_status(new_boxes, empty_status=status).value

    if not new_boxes and REGION_STATUS_INFO[status].wants_reason and rejection_reason is not None:
        doc[F.rejection_reason] = rejection_reason

    if current.get(F.status) == status.value:
        # Re-asserting the stored status (a bulk write over a mixed
        # selection) changes nothing derived from it beyond the box
        # states above: verified stays as stored. Confirming is the one
        # exception -- it is an explicit verification.
        if status == CONFIRM_STATUS:
            doc[F.verified] = True
    else:
        doc[F.verified] = status == CONFIRM_STATUS
    return doc


def reason_only_box_write(current: dict[str, Any], reason: str | None) -> dict[str, Any]:
    """W8-cleanup N3: a reason-only PATCH (no status change) over the
    current box list.

    Patches every currently-rejected box's ``rejection_reason``, same as
    ``human_status_box_write``'s per-status reason update. For a box-less
    item (``no_region_visible``), there is no rejected box for
    :func:`~src.services.curation.region_boxes.boxes_write_fields` to
    derive a mirror reason from, so it always clears the item-level
    ``rejection_reason`` to ``None`` -- restore the reason this request
    is setting in that case.

    W8-cleanup R3-1: that restore must only fire for a truly box-less
    item (``not new_boxes``). A looser "no rejected box" condition also
    matched an accepted-only, false-positive-only, or proposed-only item,
    and stored a reason on it.
    """
    F = get_region_fields()
    boxes = read_boxes(current, F)
    new_boxes = [
        dataclasses.replace(b, rejection_reason=reason) if b.state == 'rejected' else b
        for b in boxes
    ]
    doc = dict(boxes_write_fields(new_boxes, current_src=current))
    if not new_boxes:
        doc[F.rejection_reason] = reason
    return doc


def post_write_item(
    current: dict[str, Any], update: dict[str, Any], crop_id: str
) -> dict[str, Any]:
    """The wire item as stored after ``update`` is merged onto ``current``."""
    return serialize_item({**current, **update}, crop_id)


__all__ = [
    'human_box_write',
    'human_status_box_write',
    'label_source_is_human',
    'parent_to_source_bbox',
    'post_write_item',
    'reason_only_box_write',
]
