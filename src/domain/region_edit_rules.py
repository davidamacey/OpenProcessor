"""The box-level rules of a person's region edit, pure, over the boxes the person saw.

Mirrors ``src/services/curation/region_box_edits.py`` (``with_state``, ``boxes_with_status``,
``same_box``, ``apply_put_boxes``), which the writers of today call; the lane that replaces those
writers deletes that copy and imports this one. Until then the differential test in
``tests/domain/test_fold_differential.py`` keeps the two equal.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from src.config.region_rejection import (
    REJECT_REASON_HUMAN,
    REJECT_REASON_NO_VERDICT,
    REJECT_REASON_VERIFIER,
)
from src.config.region_state import RegionStatus
from src.domain.region_boxes import RegionBox, RegionBoxWriteError, next_box_id


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.domain.claim_payloads import Box4

BOX_MATCH_TOLERANCE = 1e-4
"""Max per-coordinate difference for a requested box to count as the box shown."""

_FALSE_POSITIVE = RegionStatus.FALSE_POSITIVE.value


def same_box(a: Sequence[float], b: Sequence[float], *, tol: float = BOX_MATCH_TOLERANCE) -> bool:
    return len(a) == len(b) == 4 and all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def with_state(box: RegionBox, state: str, *, rejection_reason: str | None = None) -> RegionBox:
    """One state transition: a reject carries the reason (the human one by default, which also
    marks the box a person's); every other state clears it; re-asserting a non-rejected state
    changes nothing."""
    if state == box.state and state != 'rejected':
        return box
    reason = (rejection_reason or REJECT_REASON_HUMAN) if state == 'rejected' else None
    return dataclasses.replace(box, state=state, rejection_reason=reason)


def _verifier_rejected(box: RegionBox) -> bool:
    return box.state == 'rejected' and box.rejection_reason in (
        REJECT_REASON_VERIFIER,
        REJECT_REASON_NO_VERDICT,
    )


def boxes_with_status(
    status: str, boxes: Sequence[RegionBox], *, rejection_reason: str | None = None
) -> list[RegionBox]:
    """A whole-set verdict over the list. A confirm settles only the undecided boxes (or, with
    none, reopens the verifier's rejects); it never overrides a per-box decision."""
    if status == RegionStatus.DETECTED.value:
        if not boxes:
            raise RegionBoxWriteError('no_boxes')
        result = [with_state(b, 'accepted') if b.state == 'proposed' else b for b in boxes]
        if not any(b.state == 'accepted' for b in result):
            result = [with_state(b, 'accepted') if _verifier_rejected(b) else b for b in boxes]
        if not any(b.state == 'accepted' for b in result):
            raise RegionBoxWriteError('no_accepted_box')
        return result
    if status == RegionStatus.FALSE_POSITIVE.value:
        return [with_state(b, _FALSE_POSITIVE) for b in boxes]
    if status == RegionStatus.VERIFY_REJECTED.value:
        return [with_state(b, 'rejected', rejection_reason=rejection_reason) for b in boxes]
    if status == RegionStatus.NO_REGION_VISIBLE.value:
        return []
    raise RegionBoxWriteError(f'unsupported whole-set status: {status!r}')


@dataclasses.dataclass(frozen=True)
class BoxEdit:
    """One element of a person's full-list box edit (``PUT /crops/{id}/regions``)."""

    box_id: str | None = None
    bbox: Box4 | None = None
    state: str | None = None
    text: str | None = None


def apply_edit(
    shown: Sequence[RegionBox], edits: Sequence[BoxEdit], *, box_seq: int
) -> tuple[list[RegionBox], dict[str, str]]:
    """The list a full-list edit produces, and ``{new box id: ...}`` for the boxes it made.

    An element with only ``box_id`` keeps the box; a ``bbox`` moves it (within the tolerance it is
    a confirmation, not an edit); ``state`` and ``text`` set those. ``box_id`` None is a new box,
    accepted unless a state is given. A shown box left out is deleted.
    """
    existing = {b.box_id: b for b in shown}
    result: list[RegionBox] = []
    made: dict[str, str] = {}
    seen: set[str] = set()
    for edit in edits:
        if edit.box_id is None:
            if edit.bbox is None:
                raise RegionBoxWriteError('bbox_required: a new box needs a bbox_norm')
            new_id = next_box_id([*existing.values(), *result], seq=box_seq)
            box = RegionBox(box_id=new_id, bbox_norm=edit.bbox, state='accepted', source='human')
            made[new_id] = new_id
        else:
            if edit.box_id in seen:
                raise RegionBoxWriteError(f'duplicate_box_id: {edit.box_id!r}')
            seen.add(edit.box_id)
            if edit.box_id not in existing:
                raise RegionBoxWriteError(f'unknown box_id: {edit.box_id!r}')
            box = existing[edit.box_id]
            if edit.bbox is not None and not same_box(edit.bbox, box.bbox_norm):
                box = dataclasses.replace(box, bbox_norm=edit.bbox, source='human')
        if edit.state is not None:
            box = with_state(box, edit.state)
        if edit.text is not None:
            box = dataclasses.replace(box, text=edit.text, text_source='human')
        result.append(box)
    return result, made


def box_changes(before: RegionBox | None, after: RegionBox) -> dict[str, Any]:
    """What a person changed on a box, by aspect: ``bbox`` (moved), ``state`` (state or reason),
    ``text`` (typed). A new box reports every aspect it was made with."""
    out: dict[str, Any] = {}
    if before is None or not same_box(before.bbox_norm, after.bbox_norm):
        out['bbox'] = after.bbox_norm
    if before is None or (before.state, before.rejection_reason) != (
        after.state,
        after.rejection_reason,
    ):
        out['state'] = (after.state, after.rejection_reason)
    if after.text_source == 'human' and (
        before is None or before.text_source != 'human' or before.text != after.text
    ):
        out['text'] = after.text or ''
    return out
