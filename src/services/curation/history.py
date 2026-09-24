"""Per-doc class/region history append helpers for the items index.

Every write that changes ``class_id`` / ``class_source`` / ``class_name``
should append an entry to the nested ``class_id_history`` array so we
can answer "who labeled this and when" after a model-drift investigation.

The mapping for ``class_id_history`` lives in
``src/clients/curation_opensearch.py``. Writers use the helpers here.

For region-of-interest writes, the ``RegionFields.detector_chain``
keyword-array field already captures per-event provenance; this module
owns its entry format (``<actor>:<event>``, see :func:`region_chain_entry`)
and the merge used by every writer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


MAX_HISTORY_ENTRIES = 32
"""Cap class_id_history length to prevent unbounded growth on hot crops.
Drop-oldest policy: when an append would exceed this, the earliest entry
(after the seed_backfill stub, if present) is removed.

The cap does NOT apply once the crop is ``class_validated=true`` — a
human-validated crop's audit trail is authoritative and must never be
truncated. The check reads ``class_validated`` off the pre-write
``current_source`` passed to :func:`record_class_history`."""

MAX_REGION_CHAIN_ENTRIES = 16
"""Cap the region detector-chain length. The cascade typically emits 2-4
entries per detection (e.g. detector:hit, secondary:hit, verifier:verify_ok);
16 entries covers ~4 re-detection events worth of history."""


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec='seconds')


def record_class_history(
    current_source: dict[str, Any],
    *,
    writer: str,
    now: str | None = None,
) -> list[dict[str, Any]]:
    """Build the next ``class_id_history`` array given the current doc.

    Snapshots the current class assignment as a history entry; returns
    the new array to write back. The caller integrates this into the
    OCC merger so the append happens inside the same OCC update as the
    new class write.

    Args:
        current_source: The pre-write OpenSearch ``_source`` dict.
            Reads ``class_id``, ``class_name``, ``class_source``,
            ``label_source``, ``confidence`` from it.
        writer: Identifier for the new history entry's writer
            (``ingest``, ``human``, ``vlm_pipeline``, ``sam_worker``,
            ``seed_backfill``, etc.).
        now: Optional ISO-8601 timestamp; defaults to UTC now.

    Returns:
        The new ``class_id_history`` array. If the current source has
        no class_id (first label of a brand-new crop), returns the
        existing history unchanged (nothing to preserve).
    """
    current_class_id = current_source.get('class_id')
    if current_class_id is None:
        # Nothing to preserve — first labeling of this crop. Return
        # any existing history unchanged.
        return list(current_source.get('class_id_history') or [])

    history = list(current_source.get('class_id_history') or [])
    entry = {
        'class_id': int(current_class_id),
        'class_name': current_source.get('class_name'),
        'class_source': current_source.get('class_source'),
        'label_source': current_source.get('label_source'),
        'confidence': current_source.get('confidence'),
        'writer': writer,
        'at': now or _now_iso(),
    }
    # Dedupe: skip the append when class_id AND class_source are both
    # unchanged from the last recorded entry. Without this, a writer
    # that re-asserts an unchanged class (e.g. a re-run pipeline pass
    # landing on the same conclusion, or a region-only write that
    # re-threads the existing class fields through the same merger)
    # pads the array with identical entries that carry no audit signal.
    # label_source/confidence deltas alone do NOT count as a change —
    # only class_id + class_source.
    if history:
        last = history[-1]
        if (
            last.get('class_id') == entry['class_id']
            and last.get('class_source') == entry['class_source']
        ):
            return history
    history.append(entry)
    # Drop-oldest, preserving any seed_backfill stub at index 0 if present
    # so we keep the original origin marker. Skipped entirely once
    # class_validated=true.
    return _cap(history, current_source)


HUMAN_DISCARD_WRITER = 'human:discard_crop'

HUMAN_LABEL_WRITERS = frozenset(
    {
        'human:label_crop',
        'human:batch_label_crops',
        'human:move_crops',
        'human:resolve_new_class',
        HUMAN_DISCARD_WRITER,
    }
)
"""Writers whose write the labeler's Undo (``POST /crops/{id}/label/undo``)
reverses."""

REVIEW_DISMISS_FIELDS: tuple[str, ...] = ('review_dismissed_at', 'review_dismissed_by')
"""Recorded only by the discard writer (the one human write that can change
them), and restored only from an entry that carries them."""

HUMAN_UNLABEL_WRITER = 'human:unlabel_crop'

CLASS_STATE_FIELDS: tuple[str, ...] = (
    'class_id',
    'class_name',
    'class_source',
    'label_source',
    'confidence',
    'class_detector',
    'class_detector_version',
    'class_labeler',
    'class_labeled_at',
    'class_validated',
    'cluster_id',
    'cluster_subid',
)
"""The item's full class/label/provenance/cluster state. A restorable
history entry carries every one of these so an undo can put the item back
exactly, rather than re-deriving provenance it never recorded."""


def _cap(history: list[dict[str, Any]], current_source: dict[str, Any]) -> list[dict[str, Any]]:
    if len(history) > MAX_HISTORY_ENTRIES and not current_source.get('class_validated'):
        if history and history[0].get('writer') == 'seed_backfill':
            return [history[0], *history[-(MAX_HISTORY_ENTRIES - 1) :]]
        return history[-MAX_HISTORY_ENTRIES:]
    return history


def record_class_snapshot(
    current_source: dict[str, Any],
    *,
    writer: str,
    restorable: bool,
    now: str | None = None,
) -> list[dict[str, Any]]:
    """Append a full :data:`CLASS_STATE_FIELDS` snapshot of the pre-write doc.

    Unlike :func:`record_class_history` this always appends — even for an
    item with no class yet (an ingest proposal) and even when class_id /
    class_source are unchanged — because the labeler's Undo pairs label
    writes with unlabel writes by position in the array
    (:func:`find_undo_snapshot`); a skipped append would pair the wrong
    entries. Absent fields are recorded as ``None``.

    ``restorable=True`` marks an entry written by a human label write
    (the state that write replaced); the unlabel writer records its own
    snapshot with ``restorable=False`` for the audit trail.
    """
    history = list(current_source.get('class_id_history') or [])
    entry: dict[str, Any] = {f: current_source.get(f) for f in CLASS_STATE_FIELDS}
    if entry['class_id'] is not None:
        entry['class_id'] = int(entry['class_id'])
    entry['class_validated'] = bool(entry['class_validated'])
    if writer == HUMAN_DISCARD_WRITER:
        entry.update({f: current_source.get(f) for f in REVIEW_DISMISS_FIELDS})
    entry['restorable'] = restorable
    entry['writer'] = writer
    entry['at'] = now or _now_iso()
    history.append(entry)
    return _cap(history, current_source)


def find_undo_snapshot(history: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Return the entry recording the state before the most recent
    not-yet-undone human label write, or ``None`` when there is none.

    Walks newest-first treating the history as a stack: each unlabel
    entry cancels the next older human label entry, so repeated undos
    step back through successive human labels instead of re-applying the
    same one.
    """
    pending_undos = 0
    for entry in reversed(history or []):
        writer = entry.get('writer')
        if writer == HUMAN_UNLABEL_WRITER:
            pending_undos += 1
        elif writer in HUMAN_LABEL_WRITERS:
            if pending_undos == 0:
                return entry
            pending_undos -= 1
    return None


def restore_class_state(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Update-doc fields restoring the class state recorded in ``entry``.

    ``None`` (no human label write on record) restores an unlabeled item:
    class and every provenance field cleared, nothing invented.

    Entries written before full snapshots existed carry only
    class_id/class_name/class_source/label_source/confidence; the
    remaining fields restore as ``None`` / unvalidated rather than being
    guessed.

    Cluster placement mirrors the label writers' ``cluster_id = class_id``
    rule: a restored validated class sits in its class cluster (keeping
    the recorded sub-cluster only if it belonged to that cluster);
    anything else returns to the cluster it was in before the label write
    (``None`` = residual pool).
    """
    snap = entry or {}
    out: dict[str, Any] = {f: snap.get(f) for f in CLASS_STATE_FIELDS}
    out['class_validated'] = bool(out['class_validated'])
    if entry is None:
        out['label_source'] = ''
    if out['class_validated'] and out['class_id'] is not None:
        if out['cluster_id'] != out['class_id']:
            out['cluster_subid'] = None
        out['cluster_id'] = out['class_id']
    for f in REVIEW_DISMISS_FIELDS:
        if f in snap:
            out[f] = snap[f]
    return out


def region_chain_entry(actor: str, event: str) -> str:
    """Format one ``RegionFields.detector_chain`` entry: ``<actor>:<event>``.

    ``actor`` is a detector / segmenter / verifier name (or ``vlm_visible``,
    ``human``); ``event`` may itself carry colon-separated detail
    (``sanity_reject:aspect``). No version and no timestamp: the chain is
    matched with exact ``term`` queries (``regions.py`` training-candidate
    cohorts), and ``region_detected_at`` / ``region_verified_at`` already
    carry the times.
    """
    return f'{actor}:{event}'


def normalize_region_chain_entry(entry: str) -> str:
    """Rewrite a pre-fix ``<actor>::<event>@<iso>`` entry to ``<actor>:<event>``.

    Earlier worker builds stamped an empty version slot and a timestamp
    onto every entry, which no exact-match reader could hit. Normalizing
    on merge heals a doc's chain the next time the worker writes it.
    """
    at = entry.rfind('@')
    if at > 0 and entry[at + 1 : at + 2].isdigit():
        entry = entry[:at]
    return entry.replace('::', ':', 1)


def merge_region_chain(existing: list[str] | None, new_entries: list[str] | None) -> list[str]:
    """Ordered, de-duplicated union of ``existing`` and ``new_entries``.

    Every entry is normalized to ``<actor>:<event>`` first, so a re-write
    of the same cascade outcome never grows the chain. Capped at
    :data:`MAX_REGION_CHAIN_ENTRIES`, dropping the oldest.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for raw in [*(existing or []), *(new_entries or [])]:
        entry = normalize_region_chain_entry(str(raw))
        if entry in seen:
            continue
        seen.add(entry)
        merged.append(entry)
    if len(merged) > MAX_REGION_CHAIN_ENTRIES:
        merged = merged[-MAX_REGION_CHAIN_ENTRIES:]
    return merged


__all__ = [
    'CLASS_STATE_FIELDS',
    'HUMAN_DISCARD_WRITER',
    'HUMAN_LABEL_WRITERS',
    'HUMAN_UNLABEL_WRITER',
    'MAX_HISTORY_ENTRIES',
    'MAX_REGION_CHAIN_ENTRIES',
    'REVIEW_DISMISS_FIELDS',
    'find_undo_snapshot',
    'merge_region_chain',
    'normalize_region_chain_entry',
    'record_class_history',
    'record_class_snapshot',
    'region_chain_entry',
    'restore_class_state',
]
