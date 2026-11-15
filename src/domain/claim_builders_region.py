"""Box, region and geometry edits as claims (15a sections 3.2 and 3.3, B3).

A person's region edit produces a new box list (:mod:`src.domain.region_edit_rules`); the claims
are that list read against the boxes the person saw: a box made, moved, given a state or a text,
or deleted. A whole-set verdict is written out box by box, in the same operation, so the set a
person closed is entirely claims and no machine write can reopen it. ``no_region_visible``
deletes every box and adds the one ``region_status_set``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.domain.claim_builders import HUMAN_ACTOR, ItemView, human_draft
from src.domain.claim_kinds import ActorKind, ClaimKind, SubjectKind
from src.domain.claim_payloads import (
    BoxCreated,
    BoxDeleted,
    BoxMoved,
    BoxStateSet,
    BoxTextSet,
    ClassProfile,
    ClassSet,
    GeometrySet,
    HoldoutFrozen,
    ItemDrawn,
    MachineClass,
    RegionStatusSet,
)
from src.domain.claims import ClaimDraft, Subject
from src.domain.item_geometry import validate_item_box
from src.domain.region_boxes import RegionBox
from src.domain.region_edit_rules import BoxEdit, apply_edit, box_changes, boxes_with_status


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.domain.claim_payloads import Box4


@dataclass(frozen=True)
class RegionWrite:
    drafts: list[ClaimDraft]
    box_seq: int
    """The item's ``box_seq`` after the write: it never decreases, so an id is never reused."""


def _box_id_number(box_id: str) -> int:
    digits = box_id[1:] if box_id.startswith('b') else ''
    return int(digits) if digits.isdigit() else 0


def _subject(view: ItemView, box_id: str, bbox: Box4) -> Subject:
    return Subject(
        SubjectKind.BOX,
        photo_key=view.photo_key,
        crop_id=view.crop_id,
        box_id=box_id,
        bbox=bbox,
        frame=view.frame,
    )


def _box_draft(
    view: ItemView, box_id: str, bbox: Box4, kind: ClaimKind, payload: object, op_id: str | None
) -> ClaimDraft:
    return ClaimDraft(
        ActorKind.HUMAN,
        HUMAN_ACTOR,
        op_id,
        kind,
        _subject(view, box_id, bbox),
        payload,  # type: ignore[arg-type]
    )


def _claims_for_result(
    view: ItemView,
    shown: Sequence[RegionBox],
    result: Sequence[RegionBox],
    *,
    region_seq: int,
    op_id: str | None,
    every_state: bool,
    verdict_status: str = 'no_region_visible',
) -> list[ClaimDraft]:
    by_id = {b.box_id: b for b in shown}
    drafts: list[ClaimDraft] = []
    for box in result:
        before = by_id.get(box.box_id)
        changes = box_changes(before, box)
        if before is None:
            created = BoxCreated(
                item_bbox=view.bbox,
                bbox=box.bbox_norm,
                state=box.state,
                text=box.text if box.text_source == 'human' else None,
                region_seq=region_seq,
            )
            drafts.append(
                _box_draft(view, box.box_id, box.bbox_norm, ClaimKind.BOX_CREATED, created, op_id)
            )
            continue
        if 'bbox' in changes:
            moved = BoxMoved(item_bbox=view.bbox, bbox=box.bbox_norm, region_seq=region_seq)
            drafts.append(
                _box_draft(view, box.box_id, before.bbox_norm, ClaimKind.BOX_MOVED, moved, op_id)
            )
        if 'state' in changes or every_state:
            state = BoxStateSet(
                item_bbox=view.bbox,
                state=box.state,
                rejection_reason=box.rejection_reason if box.state == 'rejected' else None,
                region_seq=region_seq,
            )
            drafts.append(
                _box_draft(view, box.box_id, box.bbox_norm, ClaimKind.BOX_STATE_SET, state, op_id)
            )
        if 'text' in changes:
            text = BoxTextSet(item_bbox=view.bbox, text=box.text or '', region_seq=region_seq)
            drafts.append(
                _box_draft(view, box.box_id, box.bbox_norm, ClaimKind.BOX_TEXT_SET, text, op_id)
            )
    kept = {b.box_id for b in result}
    drafts += [
        _box_draft(
            view,
            b.box_id,
            b.bbox_norm,
            ClaimKind.BOX_DELETED,
            BoxDeleted(item_bbox=view.bbox, region_seq=region_seq),
            op_id,
        )
        for b in shown
        if b.box_id not in kept
    ]
    if not result:
        verdict = RegionStatusSet(
            item_bbox=view.bbox,
            status=verdict_status,
            label_source='human',
            region_seq=region_seq,
        )
        drafts.append(human_draft(ClaimKind.REGION_STATUS_SET, view, verdict, op_id=op_id))
    return drafts


def _verdict(status: str | None) -> str:
    """The status a verdict over a set with no box records: the one the person said (Q7)."""
    return status if status in ('verify_rejected', 'false_positive') else 'no_region_visible'


def _write(
    view: ItemView,
    shown: Sequence[RegionBox],
    result: Sequence[RegionBox],
    *,
    box_seq: int,
    region_seq: int,
    op_id: str | None,
    every_state: bool,
    verdict_status: str = 'no_region_visible',
) -> RegionWrite:
    # An edit that leaves no box proposed closes the set: every box gets a state claim, so the
    # closed set is entirely claims and a later region re-run cannot change what the person saw.
    closes = bool(result) and not any(b.state == 'proposed' for b in result)
    drafts = _claims_for_result(
        view,
        shown,
        result,
        region_seq=region_seq,
        op_id=op_id,
        every_state=every_state or closes,
        verdict_status=verdict_status,
    )
    top = max([box_seq, *(_box_id_number(b.box_id) for b in result)])
    return RegionWrite(drafts, top)


def put_regions(
    view: ItemView,
    shown: Sequence[RegionBox],
    edits: Sequence[BoxEdit],
    *,
    box_seq: int,
    region_seq: int,
    op_id: str | None,
    whole_set_status: str | None = None,
    rejection_reason: str | None = None,
) -> RegionWrite:
    """``PUT /crops/{id}/regions`` and ``PUT /crops/batch_regions``: the full list, as a diff.
    With ``whole_set_status`` the verdict is applied to the built list in the same write."""
    result, _made = apply_edit(shown, edits, box_seq=box_seq)
    if whole_set_status is not None:
        result = boxes_with_status(whole_set_status, result, rejection_reason=rejection_reason)
    return _write(
        view,
        shown,
        result,
        box_seq=box_seq,
        region_seq=region_seq,
        op_id=op_id,
        every_state=whole_set_status is not None,
        verdict_status=_verdict(whole_set_status),
    )


def patch_box(
    view: ItemView,
    shown: Sequence[RegionBox],
    box_id: str,
    *,
    state: str | None = None,
    text: str | None = None,
    box_seq: int,
    region_seq: int,
    op_id: str | None,
) -> RegionWrite:
    """``PATCH /crops/{id}/regions/{box_id}``: one box's state or text, every sibling untouched."""
    edits = [
        BoxEdit(box_id=b.box_id, state=state, text=text)
        if b.box_id == box_id
        else BoxEdit(b.box_id)
        for b in shown
    ]
    return put_regions(view, shown, edits, box_seq=box_seq, region_seq=region_seq, op_id=op_id)


def batch_box_state(
    view: ItemView,
    shown: Sequence[RegionBox],
    box_ids: Sequence[str],
    state: str,
    *,
    box_seq: int,
    region_seq: int,
    op_id: str | None,
) -> RegionWrite:
    """``POST /regions/batch_box_state``: one state on the named boxes of one item."""
    edits = [BoxEdit(b.box_id, state=state if b.box_id in box_ids else None) for b in shown]
    return put_regions(view, shown, edits, box_seq=box_seq, region_seq=region_seq, op_id=op_id)


def set_region_status(
    view: ItemView,
    shown: Sequence[RegionBox],
    status: str,
    *,
    rejection_reason: str | None = None,
    box_seq: int,
    region_seq: int,
    op_id: str | None,
) -> RegionWrite:
    """``PATCH /crops/{id}/region_meta`` and ``POST /regions/batch_status``: a whole-set verdict,
    one state claim for every box of the resulting list (B3)."""
    result = boxes_with_status(status, shown, rejection_reason=rejection_reason)
    return _write(
        view,
        shown,
        result,
        box_seq=box_seq,
        region_seq=region_seq,
        op_id=op_id,
        every_state=True,
        verdict_status=_verdict(status),
    )


def set_rejection_reason(
    view: ItemView,
    shown: Sequence[RegionBox],
    reason: str | None,
    *,
    box_seq: int,
    region_seq: int,
    op_id: str | None,
) -> RegionWrite:
    """A reason-only ``region_meta`` patch: every rejected box gets the person's reason."""
    import dataclasses

    result = [
        dataclasses.replace(b, rejection_reason=reason) if b.state == 'rejected' else b
        for b in shown
    ]
    return _write(
        view, shown, result, box_seq=box_seq, region_seq=region_seq, op_id=op_id, every_state=False
    )


def set_geometry(
    view: ItemView,
    bbox: Sequence[float],
    *,
    drop: Sequence[RegionBox] = (),
    region_seq: int,
    op_id: str,
) -> list[ClaimDraft]:
    """``PATCH /crops/{id}/geometry``: the item's new box, and a delete for each region box the
    new box would leave outside it that the person agreed to drop. Raises ``ItemBoxError``."""
    new_box = validate_item_box(bbox)
    drafts = [human_draft(ClaimKind.GEOMETRY_SET, view, GeometrySet(bbox=new_box), op_id=op_id)]
    drafts += [
        _box_draft(
            view,
            b.box_id,
            b.bbox_norm,
            ClaimKind.BOX_DELETED,
            BoxDeleted(item_bbox=view.bbox, region_seq=region_seq),
            op_id,
        )
        for b in drop
    ]
    return drafts


def draw_object(
    view: ItemView,
    *,
    class_name: str,
    class_id_hint: int | None,
    request_id: str,
    freeze: tuple[str, MachineClass | None] | None = None,
    op_id: str,
    profile: ClassProfile | None = None,
) -> list[ClaimDraft]:
    """``POST /images/{image_id}/items``: a drawn box becomes an item, labelled by the person. On a
    held-out photo the new item is held out too, frozen as it was drawn (``freeze`` is the
    sibling's freeze id and the machine class: none, for a drawn item)."""
    new_box = validate_item_box(view.bbox)
    drafts = [
        human_draft(
            ClaimKind.ITEM_DRAWN,
            view,
            ItemDrawn(bbox=new_box, request_id=request_id),
            op_id=op_id,
        ),
        human_draft(
            ClaimKind.CLASS_SET,
            view,
            ClassSet(
                class_name=class_name,
                class_id_hint=class_id_hint,
                label_source='human',
                via='single',
                labeler='human',
                profile=profile or ClassProfile(),
            ),
            op_id=op_id,
        ),
    ]
    if freeze is not None:
        payload = HoldoutFrozen(freeze_id=freeze[0], machine_class=freeze[1])
        drafts.append(human_draft(ClaimKind.HOLDOUT_FROZEN, view, payload, op_id=op_id))
    return drafts


__all__ = [
    'BoxEdit',
    'RegionBox',
    'RegionWrite',
    'batch_box_state',
    'draw_object',
    'patch_box',
    'put_regions',
    'set_geometry',
    'set_region_status',
    'set_rejection_reason',
]
