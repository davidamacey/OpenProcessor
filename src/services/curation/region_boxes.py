"""W8 per-item region-box list: the single storage abstraction.

Regions are always a list per item. N=1 is a list of one — there is no
separate "single box" code path (see
``docs/design/openprocessor_internal/any_domain_plan.md`` W8.0-W8.2).

This module is pure (no I/O): every OpenSearch reader/writer routes
through :func:`read_boxes` / :func:`boxes_write_fields` so the list, the
item-level summary fields (``region_count``, ``region_revision``, …) and
the nested query builder (:func:`box_query`) can never disagree.

Scope note: this pass implements the core primitives only (storage
element, read/write, id assignment, status derivation, nested-query
helpers). It does NOT wire the worker pipeline, VLM overlay, human edit
routes, migration or embeddings through this module yet — see the final
handback report for the full list of what remains.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Protocol

from src.config.region_fields import RegionFields, get_region_fields
from src.config.region_rejection import REJECT_REASON_HUMAN
from src.config.region_state import RegionStatus


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


BOX_STATES: tuple[str, ...] = (
    'proposed',
    'accepted',
    'rejected',
    RegionStatus.FALSE_POSITIVE.value,
)


@dataclass(frozen=True)
class RegionBox:
    """One element of the ``region_boxes`` nested list.

    Element keys are fixed strings (not ``RegionFields``-indirected) —
    see W8.2: the list is new, so no deployment has legacy names for
    them.
    """

    box_id: str
    bbox_norm: tuple[float, float, float, float]
    state: str
    score: float | None = None
    detector: str | None = None
    detector_version: str | None = None
    source: str | None = None
    bbox_correct: bool | None = None
    confidence: str | None = None
    rejection_reason: str | None = None
    text: str | None = None
    text_raw: str | None = None
    text_source: str | None = None
    text_engine_version: str | None = None
    text_confidence: float | None = None
    text_vlm: str | None = None
    text_ocr: str | None = None
    text_choice: str | None = None
    text_vlm_invalid: str | None = None
    text_disagreement: bool | None = None
    cluster_id: int | None = None
    cluster_subid: str | None = None
    cluster_distance: float | None = None
    detected_at: str | None = None

    def to_doc(self) -> dict[str, Any]:
        """The nested-list element dict, as written to OpenSearch."""
        doc = {f.name: getattr(self, f.name) for f in fields(self)}
        doc['bbox_norm'] = list(doc['bbox_norm'])
        return doc

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> RegionBox:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in doc.items() if k in known}
        bbox = kwargs.get('bbox_norm')
        if bbox is not None:
            kwargs['bbox_norm'] = tuple(bbox)
        return cls(**kwargs)


def read_boxes(src: dict[str, Any], F: RegionFields | None = None) -> list[RegionBox]:
    """Every OpenSearch reader goes through this. Order as stored."""
    F = F or get_region_fields()
    raw = src.get(F.boxes) or []
    return [RegionBox.from_doc(d) for d in raw]


def accepted(boxes: Sequence[RegionBox]) -> list[RegionBox]:
    return [b for b in boxes if b.state == 'accepted']


class _HasBoxId(Protocol):
    """Structural type for :func:`next_box_id` -- anything naming a
    ``box_id`` (a real :class:`RegionBox`, or a lightweight id-only stand-in
    a caller builds while assigning ids to a batch of fresh candidates).

    ``box_id`` is a ``@property`` (not a plain attribute) so a concrete
    class's ``box_id: str`` field satisfies it covariantly -- mypy checks
    a plain Protocol attribute invariantly, which a `list[Concrete]` built
    from several concrete classes never satisfies.
    """

    @property
    def box_id(self) -> str: ...


def next_box_id(existing: Iterable[_HasBoxId], *, seq: int) -> str:
    """``b{max(region_box_seq, max existing id) + 1}``.

    Never reused within an item, including after deletes — the
    high-water mark (``seq``) only ever grows, it is not derived solely
    from what currently exists.
    """
    max_existing = 0
    for b in existing:
        if b.box_id and b.box_id.startswith('b'):
            try:
                max_existing = max(max_existing, int(b.box_id[1:]))
            except ValueError:
                continue
    return f'b{max(seq, max_existing) + 1}'


def derive_status(boxes: Sequence[RegionBox], *, empty_status: RegionStatus) -> RegionStatus:
    """Item status from the box list, in W8.7's fixed precedence order."""
    if any(b.state == 'accepted' for b in boxes):
        return RegionStatus.DETECTED
    if any(b.state == RegionStatus.FALSE_POSITIVE.value for b in boxes):
        return RegionStatus.FALSE_POSITIVE
    if any(b.state == 'proposed' for b in boxes):
        return RegionStatus.PENDING_VERIFICATION
    if any(b.state == 'rejected' for b in boxes):
        return RegionStatus.VERIFY_REJECTED
    return empty_status


_UNCHANGED = object()


def boxes_write_fields(
    boxes: Sequence[RegionBox],
    *,
    current_src: dict[str, Any] | None = None,
    F: RegionFields | None = None,
    set_complete: Any = _UNCHANGED,
) -> dict[str, Any]:
    """The update-doc fields every box writer sets.

    ``current_src`` is the OCC-read ``_source`` every writer already
    holds; it supplies the current ``region_revision`` / ``region_box_seq``
    high-water marks (both default to 0 when absent).
    """
    F = F or get_region_fields()
    current_src = current_src or {}

    scores = [b.score for b in boxes if b.score is not None]
    current_seq = int(current_src.get(F.box_seq) or 0)
    max_id_seen = 0
    for b in boxes:
        if b.box_id and b.box_id.startswith('b'):
            try:
                max_id_seen = max(max_id_seen, int(b.box_id[1:]))
            except ValueError:
                continue

    doc: dict[str, Any] = {
        F.boxes: [b.to_doc() for b in boxes],
        F.count: sum(1 for b in boxes if b.state == 'accepted'),
        F.rejected_count: sum(1 for b in boxes if b.state == 'rejected'),
        F.max_score: max(scores) if scores else None,
        F.revision: int(current_src.get(F.revision) or 0) + 1,
        F.box_seq: max(current_seq, max_id_seen),
    }
    if set_complete is not _UNCHANGED:
        doc[F.set_complete] = set_complete
    return doc


class RegionBoxWriteError(ValueError):
    """A human box write the request can't satisfy (422)."""


def apply_put_boxes(
    current: dict[str, Any],
    requested: Sequence[dict[str, Any]],
    *,
    frame: str,
    F: RegionFields | None = None,
    project_parent_to_source: Any = None,
) -> list[RegionBox]:
    """Sibling-preserving merge for ``PUT /crops/{crop_id}/regions``.

    ``requested`` is the full list, in display order (any_domain_plan.md
    §7.7 wire-write table): an element with only ``box_id`` keeps its
    stored box untouched; one with ``box_id`` plus other keys patches
    just those keys onto the stored box; ``box_id: None`` (or omitted)
    is a new box, assigned the next id and defaulting to ``accepted``
    when ``state`` is omitted (W8 pin 2). Omitting a stored box from
    ``requested`` deletes it.

    ``project_parent_to_source(bbox, item_bbox_norm) -> list[float]`` is
    supplied by the caller for ``frame == 'parent'`` (W8 pin 1); this
    module stays pure and does no geometry itself.
    """
    F = F or get_region_fields()
    existing = {b.box_id: b for b in read_boxes(current, F)}
    seq = int(current.get(F.box_seq) or 0)
    result: list[RegionBox] = []

    for element in requested:
        box_id = element.get('box_id')
        bbox = element.get('bbox_norm')
        if bbox is not None and frame == 'parent':
            if project_parent_to_source is None:
                msg = "frame='parent' requires project_parent_to_source"
                raise RegionBoxWriteError(msg)
            bbox = project_parent_to_source(bbox, current.get(F.bbox_norm))
        if box_id is None:
            new_id = next_box_id([*existing.values(), *result], seq=seq)
            state = element.get('state') or 'accepted'
            result.append(
                RegionBox(
                    box_id=new_id,
                    bbox_norm=tuple(bbox) if bbox is not None else (0.0, 0.0, 0.0, 0.0),
                    state=state,
                    score=1.0,
                    detector='human',
                    source='human',
                    text=element.get('text'),
                )
            )
            continue
        stored = existing.get(box_id)
        if stored is None:
            msg = f'unknown box_id: {box_id!r}'
            raise RegionBoxWriteError(msg)
        patch: dict[str, Any] = {}
        if bbox is not None:
            patch['bbox_norm'] = tuple(bbox)
        if 'state' in element and element['state'] is not None:
            patch['state'] = element['state']
        if 'text' in element and element['text'] is not None:
            patch['text'] = element['text']
        result.append(stored if not patch else _replace(stored, **patch))

    return result


def _replace(box: RegionBox, **kwargs: Any) -> RegionBox:
    doc = box.to_doc()
    doc.update(kwargs)
    if 'bbox_norm' in kwargs:
        doc['bbox_norm'] = list(kwargs['bbox_norm'])
    return RegionBox.from_doc(doc)


def boxes_with_status(status: str, boxes: Sequence[RegionBox]) -> list[RegionBox]:
    """Whole-set human status transition over the list (W8.7 table).

    A whole-set confirm never overrides a per-box decision that already
    settled a box; it only settles the undecided ones.
    """
    if status == RegionStatus.DETECTED.value:
        if not boxes:
            msg = 'no_boxes'
            raise RegionBoxWriteError(msg)
        result = [_replace(b, state='accepted') if b.state == 'proposed' else b for b in boxes]
        if not any(b.state == 'accepted' for b in result):
            msg = 'no_accepted_box'
            raise RegionBoxWriteError(msg)
        return result
    if status == RegionStatus.FALSE_POSITIVE.value:
        return [_replace(b, state=RegionStatus.FALSE_POSITIVE.value) for b in boxes]
    if status == RegionStatus.VERIFY_REJECTED.value:
        return [_replace(b, state='rejected', rejection_reason=REJECT_REASON_HUMAN) for b in boxes]
    if status == RegionStatus.NO_REGION_VISIBLE.value:
        return []
    msg = f'unsupported whole-set status: {status!r}'
    raise RegionBoxWriteError(msg)


def box_query(clause: dict[str, Any], F: RegionFields | None = None) -> dict[str, Any]:
    """Wrap a per-box clause in the one nested-query shape every reader uses."""
    F = F or get_region_fields()
    return {'nested': {'path': F.boxes, 'query': clause}}


def has_any_box_query(F: RegionFields | None = None) -> dict[str, Any]:
    """``region_count >= 1`` OR ``region_rejected_count >= 1``."""
    F = F or get_region_fields()
    return {
        'bool': {
            'should': [
                {'range': {F.count: {'gte': 1}}},
                {'range': {F.rejected_count: {'gte': 1}}},
            ],
            'minimum_should_match': 1,
        }
    }


__all__ = [
    'BOX_STATES',
    'RegionBox',
    'RegionBoxWriteError',
    'accepted',
    'apply_put_boxes',
    'box_query',
    'boxes_with_status',
    'boxes_write_fields',
    'derive_status',
    'has_any_box_query',
    'next_box_id',
    'read_boxes',
]
