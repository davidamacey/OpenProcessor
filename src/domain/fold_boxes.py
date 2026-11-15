"""Layer 4 of the fold: the boxes of an item and its region status (15a section 4.3, B3).

Per box, the deciding claim of each aspect (geometry, state, text) wins over the machine box;
``box_deleted`` removes a box and never creates one. A person closes an item's box set by
leaving no box ``proposed``; machine boxes that arrive after that stay results and are not
shown, which is today's item lock expressed as data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from src.config.region_rejection import REJECT_REASON_HUMAN
from src.config.region_state import RegionStatus
from src.domain.claim_kinds import ActorKind, ClaimKind
from src.domain.claim_layers import deciding, of_kinds
from src.domain.claim_match import BoxRef, match_boxes
from src.domain.claim_payloads import BoxCreated, BoxMoved, BoxStateSet, BoxTextSet
from src.domain.fold_locks import box_is_locked
from src.domain.fold_state import BoxState
from src.domain.region_boxes import RegionBox, derive_status
from src.domain.region_text import match_fields


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.domain.claim_payloads import Box4
    from src.domain.claims import Claim
    from src.domain.fold_state import FoldSettings
    from src.domain.results import ItemInputs, MachineBox

_ASPECT_STATE = {ClaimKind.BOX_CREATED, ClaimKind.BOX_STATE_SET}
_ASPECT_GEOMETRY = {ClaimKind.BOX_CREATED, ClaimKind.BOX_MOVED}
_ASPECT_LIFE = {ClaimKind.BOX_CREATED, ClaimKind.BOX_DELETED}
_ASSERTING = {
    ClaimKind.BOX_CREATED,
    ClaimKind.BOX_MOVED,
    ClaimKind.BOX_STATE_SET,
    ClaimKind.BOX_TEXT_SET,
}


@dataclass(frozen=True)
class BoxFold:
    boxes: tuple[BoxState, ...]
    closed: bool
    status: str | None
    verifier: str | None
    set_complete: bool | None
    has_box_claims: bool
    no_region_verdict: bool
    generation: int
    """The machine generation a person sees: what their next claim's ``region_seq`` must say."""


def _seq(claim: Claim) -> int:
    seq = getattr(claim.payload, 'region_seq', None)
    return 0 if seq is None else int(seq)


def _source_of(actor: ActorKind) -> str:
    return 'human' if actor is ActorKind.HUMAN else 'import'


def _box_number(box_id: str) -> tuple[int, str]:
    digits = box_id[1:] if box_id.startswith('b') else ''
    return (int(digits), box_id) if digits.isdigit() else (1 << 60, box_id)


def _group_by_box(claims: Sequence[Claim]) -> dict[str, list[Claim]]:
    groups: dict[str, list[Claim]] = {}
    for claim in sorted(claims, key=lambda c: c.claim_id):
        if claim.subject.box_id is not None:
            groups.setdefault(claim.subject.box_id, []).append(claim)
    return groups


def _geometry_of(group: Sequence[Claim]) -> Claim | None:
    return deciding(of_kinds(group, _ASPECT_GEOMETRY))


def _claimed_bbox(claim: Claim) -> Box4:
    payload = claim.payload
    assert isinstance(payload, BoxCreated | BoxMoved)
    return payload.bbox


def _candidate_boxes(group: Sequence[Claim]) -> list[Box4 | None]:
    geometry = _geometry_of(group)
    return [None if geometry is None else _claimed_bbox(geometry), group[-1].subject.bbox]


def _state_fields(
    machine: MachineBox | None, group: Sequence[Claim]
) -> tuple[str, str | None, bool]:
    """``(state, rejection_reason, decided by a claim)``: a human reject carries the human
    reason unless the claim names another; every other state clears the reason."""
    claim = deciding(of_kinds(group, _ASPECT_STATE))
    if claim is None:
        if machine is None:
            return 'proposed', None, False
        return machine.state, machine.rejection_reason, False
    payload = claim.payload
    assert isinstance(payload, BoxCreated | BoxStateSet)
    reason = None
    if payload.state == 'rejected':
        reason = getattr(payload, 'rejection_reason', None) or REJECT_REASON_HUMAN
    return payload.state, reason, True


def _text_fields(
    machine: MachineBox | None, group: Sequence[Claim], settings: FoldSettings
) -> dict[str, object]:
    texts = [
        c
        for c in group
        if c.kind is ClaimKind.BOX_TEXT_SET
        or (c.kind is ClaimKind.BOX_CREATED and getattr(c.payload, 'text', None) is not None)
    ]
    claim = deciding(texts)
    if claim is not None:
        payload = claim.payload
        assert isinstance(payload, BoxCreated | BoxTextSet)
        text = payload.text or ''
        return {
            'text': text,
            'text_source': _source_of(claim.actor_kind),
            'text_choice': 'human',
            'text_confidence': 1.0 if text else None,
            'text_disagreement': None,
            'text_fold': match_fields(text, dict(settings.confusables))['text_fold'],
        }
    if machine is None or machine.text is None:
        return {}
    t = machine.text
    return {
        'text': t.text,
        'text_source': t.source,
        'text_choice': t.choice,
        'text_confidence': t.confidence,
        'text_disagreement': t.disagreement,
        'text_fold': t.fold,
    }


def _build_box(
    machine: MachineBox | None, group: Sequence[Claim], box_id: str, settings: FoldSettings
) -> BoxState | None:
    life = deciding(of_kinds(group, _ASPECT_LIFE))
    if life is not None and life.kind is ClaimKind.BOX_DELETED:
        return None
    state, reason, _ = _state_fields(machine, group)
    geometry = _geometry_of(group)
    fields: dict[str, object] = {}
    if geometry is not None:
        payload = geometry.payload
        fields.update(
            bbox=_claimed_bbox(geometry),
            source=_source_of(geometry.actor_kind),
            score=1.0,
            detector=settings.human_detector_name,
            detector_version=settings.human_detector_version,
            bbox_correct=payload.bbox_correct if isinstance(payload, BoxMoved) else None,
            confidence=None,
            detected_at=geometry.at.isoformat(),
        )
    elif machine is not None:
        fields.update(
            bbox=machine.bbox,
            source=machine.source,
            score=machine.score,
            detector=machine.detector,
            detector_version=machine.detector_version,
            bbox_correct=machine.bbox_correct,
            confidence=machine.confidence,
            detected_at=machine.detected_at,
        )
    else:
        fields.update(bbox=group[-1].subject.bbox)
    text = _text_fields(machine, group, settings)
    built = BoxState(
        box_id=box_id,
        ordinal=0,
        state=state,
        rejection_reason=reason,
        from_claims_only=machine is None,
        **fields,  # type: ignore[arg-type]
        **text,  # type: ignore[arg-type]
    )
    return replace(
        built,
        locked=box_is_locked(
            source=built.source,
            rejection_reason=built.rejection_reason,
            text_source=built.text_source,
            state=built.state,
        ),
    )


def _apply(
    machine: Sequence[MachineBox], claims: Sequence[Claim], settings: FoldSettings
) -> list[BoxState]:
    groups = _group_by_box(claims)
    # Question 6: a box a person made owns the id minted for it (``box_seq`` never reuses one), so
    # only claims on boxes the machine made are matched by overlap when a re-run changed the ids.
    created = {bid for bid, g in groups.items() if any(c.kind is ClaimKind.BOX_CREATED for c in g)}
    mapping = match_boxes(
        {bid: _candidate_boxes(g) for bid, g in groups.items() if bid not in created},
        [BoxRef(m.box_id, m.bbox) for m in machine],
    )
    by_machine: dict[str, list[Claim]] = {}
    for bid, mid in mapping.items():
        by_machine.setdefault(mid, []).extend(groups[bid])
    out: list[BoxState | None] = [
        _build_box(
            m, sorted(by_machine.get(m.box_id, []), key=lambda c: c.claim_id), m.box_id, settings
        )
        for m in machine
    ]
    for bid, group in groups.items():
        if bid in mapping or not any(c.kind in _ASSERTING for c in group):
            continue
        out.append(_build_box(None, group, bid, settings))
    boxes = sorted((b for b in out if b is not None), key=lambda b: _box_number(b.box_id))
    return [replace(b, ordinal=i) for i, b in enumerate(boxes)]


def _derived_status(boxes: Sequence[BoxState]) -> str:
    stubs = [RegionBox(box_id=b.box_id, bbox_norm=b.bbox, state=b.state) for b in boxes]
    return derive_status(stubs, empty_status=RegionStatus.NO_REGION_VISIBLE).value


def fold_boxes(inp: ItemInputs, live: Sequence[Claim], settings: FoldSettings) -> BoxFold:
    box_claims = of_kinds(
        live, {*_ASPECT_STATE, *_ASPECT_GEOMETRY, *_ASPECT_LIFE, ClaimKind.BOX_TEXT_SET}
    )
    verdicts = of_kinds(live, {ClaimKind.REGION_STATUS_SET})
    region_claims = [*box_claims, *verdicts]
    newest = max(region_claims, key=lambda c: c.claim_id, default=None)
    verdict = max(verdicts, key=lambda c: c.claim_id, default=None)
    newest_box = max((c.claim_id for c in box_claims), default=-1)
    no_region = verdict is not None and verdict.claim_id > newest_box

    generations = sorted({m.region_seq for m in inp.boxes})
    newest_boxes = [m for m in inp.boxes if generations and m.region_seq == generations[-1]]
    generation = generations[-1] if generations else 0
    if no_region:
        boxes: list[BoxState] = []
        closed = True
        generation = max((g for g in generations if g <= _seq(newest)), default=0) if newest else 0
    elif newest is None:
        boxes = _apply(newest_boxes, box_claims, settings)
        closed = False
    else:
        # The set the person saw is the newest machine generation not newer than their claim. A
        # region re-run adds a generation; the repository keeps the one a live box claim saw, so
        # a replaced row cannot make an open set look closed.
        seen_gen = max((g for g in generations if g <= _seq(newest)), default=None)
        seen = [m for m in inp.boxes if m.region_seq == seen_gen]
        boxes = _apply(seen, box_claims, settings)
        closed = not any(b.state == 'proposed' for b in boxes)
        if closed:
            generation = seen_gen if seen_gen is not None else 0
        else:
            boxes = _apply(newest_boxes, box_claims, settings)

    machine_region = inp.region
    if no_region:
        status: str | None = RegionStatus.NO_REGION_VISIBLE.value
    elif newest is not None:
        status = _derived_status(boxes)
    else:
        status = None if machine_region is None else machine_region.status
    verifier: str | None
    if newest is not None:
        verifier = (
            settings.human_detector_name if newest.actor_kind is ActorKind.HUMAN else 'import'
        )
    else:
        verifier = None if machine_region is None else machine_region.verifier
    set_complete = (
        True if closed else (None if machine_region is None else machine_region.set_complete)
    )
    return BoxFold(
        boxes=tuple(boxes),
        closed=closed,
        status=status,
        verifier=verifier,
        set_complete=set_complete,
        has_box_claims=newest is not None,
        no_region_verdict=no_region,
        generation=generation,
    )
