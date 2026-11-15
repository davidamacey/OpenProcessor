"""A claim between its domain value and its row (15a sections 2.2 and 3.3).

Pure: no session. ``draft_row`` is what ``INSERT`` takes, ``claim_from_row`` rebuilds the typed
:class:`~src.domain.claims.Claim` the fold reads, payload and all, through the same pydantic models
that validated it on write (:func:`src.domain.claim_payloads.parse_payload`). Boxes are stored as
``real[4]`` (float4), so a box read back differs from the double that was written by at most 6e-8;
every reader of a claim sees the stored value, so the fold is a function of the rows alone.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from src.domain.claim_kinds import ActorKind, ClaimKind, SubjectKind
from src.domain.claim_payloads import Box4, parse_payload, payload_json
from src.domain.claims import Claim, Subject
from src.domain.wire_ids import to_uuid, to_wire


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.domain.claims import ClaimDraft

# The columns a claim query selects, in one place: the loaders and the undo walk share it.
CLAIM_COLUMNS = (
    'claim_id', 'at', 'actor_kind', 'actor', 'op_id', 'kind', 'subject_kind', 'photo_key',
    'subject_crop_id', 'subject_box_id', 'subject_pair_id', 'subject_bbox', 'subject_machine_bbox',
    'subject_frame', 'withdraws_claim_id', 'payload',
)  # fmt: skip


def _box(value: list[float] | None) -> Box4 | None:
    if value is None:
        return None
    a, b, c, d = value
    return (float(a), float(b), float(c), float(d))


def draft_row(project_id: uuid.UUID, draft: ClaimDraft) -> dict[str, Any]:
    """Every column of one insert, so rows of a multi-row ``VALUES`` share their shape. ``claim_id``
    and ``at`` are the table's: the rising id and the transaction time."""
    s = draft.subject
    return {
        'project_id': project_id,
        'actor_kind': draft.actor_kind.value,
        'actor': draft.actor,
        'op_id': None if draft.op_id is None else uuid.UUID(draft.op_id),
        'kind': draft.kind.value,
        'subject_kind': s.kind.value,
        'photo_key': s.photo_key,
        'subject_crop_id': None if s.crop_id is None else to_uuid(s.crop_id),
        'subject_box_id': s.box_id,
        'subject_pair_id': None if s.pair_id is None else to_uuid(s.pair_id),
        'subject_bbox': None if s.bbox is None else list(s.bbox),
        'subject_machine_bbox': None if s.machine_bbox is None else list(s.machine_bbox),
        'subject_frame': s.frame,
        'withdraws_claim_id': draft.withdraws_claim_id,
        'payload': payload_json(draft.payload),
    }


def claim_from_row(row: Mapping[str, Any] | Any) -> Claim:
    def get(name: str) -> Any:
        return row[name] if isinstance(row, dict) else getattr(row, name)

    kind = ClaimKind(get('kind'))
    crop, pair, op = get('subject_crop_id'), get('subject_pair_id'), get('op_id')
    subject = Subject(
        SubjectKind(get('subject_kind')),
        photo_key=get('photo_key'),
        crop_id=None if crop is None else to_wire(crop),
        box_id=get('subject_box_id'),
        bbox=_box(get('subject_bbox')),
        machine_bbox=_box(get('subject_machine_bbox')),
        frame=get('subject_frame'),
        pair_id=None if pair is None else to_wire(pair),
    )
    return Claim(
        claim_id=get('claim_id'),
        at=get('at'),
        actor_kind=ActorKind(get('actor_kind')),
        actor=get('actor'),
        op_id=None if op is None else str(op),
        kind=kind,
        subject=subject,
        payload=parse_payload(kind, get('payload')),
        withdraws_claim_id=get('withdraws_claim_id'),
    )
