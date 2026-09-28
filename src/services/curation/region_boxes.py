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

import dataclasses
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Protocol

from src.config.region_fields import RegionFields, get_region_fields
from src.config.region_rejection import REJECT_REASON_HUMAN
from src.config.region_state import BOX_STATE_ROUTES, RegionStatus


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


@dataclass(frozen=True)
class _IdOnly:
    """Structural stand-in satisfying :class:`_HasBoxId` -- lets
    :func:`next_box_id` / :func:`finalize_box_ids` walk ids already spoken
    for (stored boxes plus ones assigned earlier in the same call) without
    needing a full :class:`RegionBox`."""

    box_id: str


# W8 M1 fix (pipeline-wiring review, 2026-09-27): a fresh candidate box's
# REAL id (``b<N>``) is only safe to mint against the CURRENT stored
# ``region_box_seq`` high-water mark, which is only known at write time
# (inside the bulk-writer's OCC merge, which re-reads the live doc
# immediately before writing -- see ``bulk_writer._merge``). Minting a
# real id at task-processing time off a possibly-stale fetch-time
# snapshot let a concurrent write's ids collide (the review's M1 probe).
# A placeholder is assigned instead at candidate-resolution time and
# swapped for a real id in :func:`finalize_box_ids`, called from the
# merge closure. Never matches ``b<digits>`` or a human/real box id.
_PLACEHOLDER_PREFIX = '\x00pending:'


def new_box_placeholder(index: int) -> str:
    """A temporary id for a fresh (not yet persisted) candidate box.

    ``index`` only needs to be unique within the same candidate list
    (multiple fresh boxes in one pass) -- see the module docstring above
    :data:`_PLACEHOLDER_PREFIX`.
    """
    return f'{_PLACEHOLDER_PREFIX}{index}'


def finalize_box_ids(
    boxes: Sequence[RegionBox], *, existing: Iterable[_HasBoxId], seq: int
) -> list[RegionBox]:
    """Replace every placeholder id in ``boxes`` with a real one minted
    against the CURRENT ``seq`` high-water mark (W8 M1).

    ``existing`` is the live stored box list (read fresh, immediately
    before this call) so a real id already spoken for -- by a box this
    pass never touched, including one a concurrent human write just
    added -- is never reused. Boxes that already carry a real id
    (re-verified from a stored ``proposed`` box, W8 B1) pass through
    unchanged and also claim their id against ``existing`` for the rest
    of this call's minting.
    """
    assigned: list[_HasBoxId] = list(existing)
    result: list[RegionBox] = []
    for box in boxes:
        finalized = box
        if box.box_id.startswith(_PLACEHOLDER_PREFIX):
            finalized = dataclasses.replace(box, box_id=next_box_id(assigned, seq=seq))
        assigned.append(_IdOnly(box_id=finalized.box_id))
        result.append(finalized)
    return result


def merge_boxes_for_write(
    stored: Sequence[RegionBox],
    new: Sequence[RegionBox],
    *,
    baseline: Sequence[RegionBox] | None = None,
) -> list[RegionBox]:
    """Merge this pass's resolved ``new`` boxes back into the item's full
    ``stored`` list, in stored order (W8 B1).

    A stored box this pass touched (its id appears in ``new``) is
    replaced in place; any stored box this pass never sent to the VLM --
    a sibling in another state (already accepted/rejected/false_positive,
    or a second ``proposed`` box this pass didn't select) -- is carried
    through unchanged, never silently dropped. A box in ``new`` with no
    stored counterpart (this pass's own fresh detection) is appended, in
    ``new``'s order. ``stored`` empty (the common fresh-detection case --
    no prior box list at all) is a pure replace: unchanged behaviour.

    ``baseline`` (M1 residual / R-M3 fix, 2026-09-27 re-review): the
    snapshot of ``stored`` this pass actually READ before sending its
    candidates to the VLM (``_ItemTask.stored_boxes``, fetch-time). If a
    box a candidate in ``new`` targets has since changed in ``stored``
    (a human moved it) relative to that snapshot, the human's newer
    state wins -- this pass's verdict for that box id is dropped and the
    CURRENT ``stored`` copy passes through untouched instead of being
    overwritten with geometry the VLM verified against a version that no
    longer exists. If the box has since been DELETED from ``stored``
    entirely (present in ``baseline``, absent from ``stored``), it is not
    resurrected by appending ``new``'s entry for it. Only checked for ids
    ``baseline`` actually knows about -- a box neither this pass nor
    ``baseline`` has ever seen (a genuinely fresh detection minted this
    same pass) is unaffected and appended as before. ``baseline=None``
    (the default) disables the guard entirely -- existing callers that
    never pass it keep the pre-fix behaviour.
    """
    if not stored:
        return list(new)
    baseline_by_id = {b.box_id: b for b in (baseline or ())}
    remaining = {b.box_id: b for b in new}
    merged: list[RegionBox] = []
    for s in stored:
        candidate = remaining.pop(s.box_id, None)
        if candidate is None:
            merged.append(s)
            continue
        base = baseline_by_id.get(s.box_id)
        if base is not None and base != s:
            # Human edit landed on this exact box while this pass's VLM
            # call was in flight -- keep the human's current state, drop
            # this pass's now-stale verdict for it.
            merged.append(s)
            continue
        merged.append(candidate)
    for b in new:
        if b.box_id not in remaining:
            continue
        if b.box_id in baseline_by_id:
            # Existed at fetch time, missing from `stored` now -- deleted
            # by a human during this pass. Must not be resurrected.
            continue
        merged.append(b)
    return merged


def is_human_owned(box: RegionBox) -> bool:
    """True if a human created this box OR explicitly acted on it (W8c M3).

    ``source == 'human'`` alone only covers a box a human CREATED (``PUT
    .../regions`` with ``box_id: null``). A human's per-box accept/reject
    verdict on a MACHINE-created box, via ``PATCH .../regions/{box_id}``
    or ``POST /regions/batch_box_state``, leaves ``source``/``detector``
    exactly as they were -- the only trace is the stamp those write paths
    now also set on that verdict (``rejection_reason=REJECT_REASON_HUMAN``
    for a reject, matching :func:`boxes_with_status`'s whole-set path;
    ``text_source='human'`` for a human-typed transcription). Both
    :func:`~scripts.curation.worker.bulk_writer._merge` (fresh-detection
    replace-machine/keep-human) and :func:`region_requeue.apply_requeue`
    (``clear_detection``'s box drop) key their "never a human's" guarantee
    off this, not the narrower ``source`` check alone.
    """
    return (
        box.source == 'human'
        or box.rejection_reason == REJECT_REASON_HUMAN
        or box.text_source == 'human'
    )


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


def _best(boxes: Sequence[RegionBox]) -> RegionBox | None:
    if not boxes:
        return None
    return max(boxes, key=lambda b: b.score if b.score is not None else -1.0)


def _mirror_representative(boxes: Sequence[RegionBox]) -> RegionBox | None:
    """The box the legacy ``bbox_norm``/``score``/``detector``/... mirror
    fields describe (W8-cleanup M2c): the highest-scoring
    accepted-or-false_positive box, if any -- the *best* one, not the
    first one written.

    Deliberately never a rejected box: :mod:`region_boxes`'s own module
    docstring says a box in ``candidate_bbox_norm`` (rejected) is
    NOT ``bbox_norm``, because ``bbox_norm`` is an accepted region to
    every reader (browse, export, clustering). Falling back to a
    rejected box's coordinates here would make a rejected box look
    accepted to all of them.
    """
    return _best([b for b in boxes if b.state in ('accepted', RegionStatus.FALSE_POSITIVE.value)])


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

    Also (re)computes the legacy per-item mirror fields (``bbox_norm``,
    ``score``, ``detector``, ``detector_version``, ``source``,
    ``bbox_frame``, ``rejection_reason``) from :func:`_mirror_representative`
    on *every* call, and clears the retired ``candidate_*`` fields --
    W8-cleanup M2's fix for the mirror going stale on any writer that
    isn't ``human_status_box_write`` (per-box PATCH, ``PUT
    .../regions``, ``POST regions/batch_box_state``, requeue, the
    worker). A caller with nothing left to mirror (empty box list) gets
    every mirror field cleared to ``None``.
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
    rep = _mirror_representative(boxes)
    if rep is not None:
        doc[F.bbox_norm] = list(rep.bbox_norm)
        doc[F.score] = rep.score
        doc[F.detector] = rep.detector
        doc[F.detector_version] = rep.detector_version
        doc[F.source] = rep.source
        doc[F.bbox_frame] = 'source'
    else:
        doc[F.bbox_norm] = None
        doc[F.score] = None
        doc[F.detector] = None
        doc[F.detector_version] = None
        doc[F.source] = None
    # `rejection_reason` mirrors the highest-scoring REJECTED box
    # independently of the accepted/FP mirror above -- unlike
    # bbox_norm/score/..., showing a rejection reason never makes a
    # rejected box look accepted to a reader.
    rejected_rep = _best([b for b in boxes if b.state == 'rejected'])
    doc[F.rejection_reason] = rejected_rep.rejection_reason if rejected_rep else None
    doc.update(
        dict.fromkeys(
            (
                F.candidate_bbox_norm,
                F.candidate_score,
                F.candidate_detector,
                F.candidate_detector_version,
                F.candidate_source,
            )
        )
    )
    if set_complete is not _UNCHANGED:
        doc[F.set_complete] = set_complete
    return doc


class RegionBoxWriteError(ValueError):
    """A human box write the request can't satisfy (422)."""


_BOX_STATE_ROUTES_BY_NAME = {r.route: r.states for r in BOX_STATE_ROUTES}


def validate_box_state(route: str, state: str | None) -> None:
    """W8c: enforce ``BOX_STATE_ROUTES`` (``src/config/region_state.py``) on
    write, not just serve it on ``GET .../regions/statuses``.

    ``route`` is the exact ``BoxStateRoute.route`` string (e.g. ``'PATCH
    /crops/{crop_id}/regions/{box_id}'``); a route this table doesn't know
    about is a programming error (``ValueError``), never a client-facing
    422. ``state=None`` (untouched / no state in this write) is always
    fine -- the caller may not be setting a state at all.
    """
    if state is None:
        return
    try:
        allowed = _BOX_STATE_ROUTES_BY_NAME[route]
    except KeyError as exc:
        msg = f'no BOX_STATE_ROUTES entry for route {route!r}'
        raise ValueError(msg) from exc
    if state not in allowed:
        msg = f'state must be one of {sorted(allowed)} for {route}; got {state!r}'
        raise RegionBoxWriteError(msg)


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
            if element['state'] == 'rejected':
                # W8c M3: a human REJECTING a machine-created box via PUT
                # (not just creating one) must also be recognized as
                # human-owned (is_human_owned) -- source/detector stay
                # whatever the machine wrote, so the reason is the only
                # trace, matching boxes_with_status's whole-set path.
                patch['rejection_reason'] = REJECT_REASON_HUMAN
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
    'finalize_box_ids',
    'has_any_box_query',
    'is_human_owned',
    'merge_boxes_for_write',
    'new_box_placeholder',
    'next_box_id',
    'read_boxes',
    'validate_box_state',
]
