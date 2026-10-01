"""Restorable snapshots for the human item edits that are not class writes.

Human class writes snapshot into ``class_id_history`` (see
:mod:`src.services.curation.history`). Region edits (confirm, reject,
false positive, box edit, status/text changes) and VLM-suggestion
dismissals snapshot here instead, into one kind-tagged list,
:data:`EDIT_HISTORY_FIELD`:

``{kind, writer, at, restorable, state: {field: value, ...}}``

Why a separate, generalized list rather than more ``class_id_history``
entries or one list per kind:

- ``class_id_history`` is a ``nested`` mapping with class-typed
  properties, is capped with a class-validated exemption, and is served
  verbatim by ``GET /crops/{id}/history`` as class state; region boxes and
  dismissal fields don't fit its schema and would evict class entries.
- Every kind here needs the same thing — snapshot the fields a write
  replaces, then undo step-wise — so one list with a ``kind`` tag shares a
  single stack walker (:func:`find_edit_undo`) and a single mapping
  (``enabled: false``: stored, never indexed; nothing queries into it).

Undo pairs entries per kind like the class undo: a write records the
pre-write state with ``restorable=True``; an undo records the state it
replaced with ``restorable=False``, which cancels the next older
restorable entry of the same kind. Repeated undos step back through
successive edits.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.config import get_region_fields


EDIT_HISTORY_FIELD = 'edit_history'

MAX_EDIT_HISTORY_ENTRIES = 64
"""Drop-oldest cap. Dropping from the old end never orphans a pairing that
matters: an undo record is always newer than the entry it cancels."""


class EditKind(str, Enum):
    REGION = 'region'
    VLM_DISMISS = 'vlm_dismiss'


REGION_UNDO_WRITER = 'human:region_undo'
VLM_DISMISS_UNDO_WRITER = 'human:vlm_dismiss_undo'

VLM_DISMISS_FIELDS: tuple[str, ...] = (
    'vlm_dismissed_class_id',
    'vlm_dismissed_class_name',
    'vlm_dismissed_at',
)


def region_state_fields() -> tuple[str, ...]:
    """Storage names of every region field a human region write can change.

    W8 pin 4: gains the per-item box-list fields (``boxes``, ``count``,
    ``rejected_count``, ``max_score``, ``set_complete``) alongside the
    pre-W8 per-box scalars, so ``POST /crops/{id}/region/undo`` restores
    whichever shape the write actually touched. The legacy scalars stay
    in this tuple (deliberately NOT dropped, unlike the AD W8.7 end-state)
    because the worker pipeline (W8b) still writes them exclusively this
    pass; dropping them here would silently stop undo from restoring a
    worker-written region. See the W8 handback report.
    """
    F = get_region_fields()
    return (
        F.bbox_norm,
        F.bbox_frame,
        F.status,
        F.score,
        F.verified,
        F.verified_at,
        F.verifier,
        F.verifier_version,
        F.validated,
        F.label_source,
        F.detector,
        F.detector_version,
        F.detected_at,
        F.source,
        F.rejection_reason,
        F.candidate_bbox_norm,
        F.candidate_score,
        F.candidate_detector,
        F.candidate_detector_version,
        F.candidate_source,
        F.text,
        F.text_source,
        F.text_confidence,
        F.text_choice,
        F.cluster_id,
        F.cluster_subid,
        F.cluster_distance,
        # W8 per-item box list (pin 4).
        F.boxes,
        F.count,
        F.rejected_count,
        F.max_score,
        F.set_complete,
    )


def state_fields(kind: EditKind) -> tuple[str, ...]:
    if kind == EditKind.REGION:
        return region_state_fields()
    return VLM_DISMISS_FIELDS


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec='seconds')


def record_edit(
    current: dict[str, Any],
    *,
    kind: EditKind,
    writer: str,
    restorable: bool = True,
    now: str | None = None,
) -> list[dict[str, Any]]:
    """The next :data:`EDIT_HISTORY_FIELD` list: ``current``'s ``kind`` state
    appended as one entry. Absent fields are recorded as ``None`` so a
    restore puts the item back exactly (absent == null in OpenSearch)."""
    history = [e for e in (current.get(EDIT_HISTORY_FIELD) or []) if isinstance(e, dict)]
    history.append(
        {
            'kind': kind.value,
            'writer': writer,
            'at': now or _now_iso(),
            'restorable': restorable,
            'state': {f: current.get(f) for f in state_fields(kind)},
        }
    )
    return history[-MAX_EDIT_HISTORY_ENTRIES:]


def find_edit_undo(history: list[Any] | None, kind: EditKind) -> dict[str, Any] | None:
    """The entry holding the state before the most recent not-yet-undone
    ``kind`` edit, or ``None`` when there is nothing to undo."""
    pending_undos = 0
    for entry in reversed(history or []):
        if not isinstance(entry, dict) or entry.get('kind') != kind.value:
            continue
        if not entry.get('restorable'):
            pending_undos += 1
        elif pending_undos == 0:
            return entry
        else:
            pending_undos -= 1
    return None


def restore_edit_state(
    entry: dict[str, Any], kind: EditKind, *, current: dict[str, Any]
) -> dict[str, Any]:
    """Update-doc fields putting back the state recorded in ``entry``.

    ``current`` is the stored doc the restore applies to. A region restore
    is itself a write over the box list, so ``region_revision`` moves
    forward from ``current`` (the stale-edit guard must see an undo as a
    change) -- a snapshot's old revision is never restored. ``region_box_seq``
    is not a snapshot field either, so the id high-water mark never
    decreases and a restored list can't collide with a box added since.
    """
    state = entry.get('state') or {}
    # Only the fields the snapshot recorded: an entry written before a
    # field joined ``state_fields`` says nothing about it, and restoring it
    # as null would erase a value that edit never touched.
    restored = {f: state[f] for f in state_fields(kind) if f in state}
    if kind == EditKind.REGION:
        F = get_region_fields()
        restored[F.revision] = int(current.get(F.revision) or 0) + 1
    return restored


__all__ = [
    'EDIT_HISTORY_FIELD',
    'MAX_EDIT_HISTORY_ENTRIES',
    'REGION_UNDO_WRITER',
    'VLM_DISMISS_FIELDS',
    'VLM_DISMISS_UNDO_WRITER',
    'EditKind',
    'find_edit_undo',
    'record_edit',
    'region_state_fields',
    'restore_edit_state',
    'state_fields',
]
