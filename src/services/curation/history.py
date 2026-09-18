"""Per-doc class/region history append helpers for the items index.

Every write that changes ``class_id`` / ``class_source`` / ``class_name``
should append an entry to the nested ``class_id_history`` array so we
can answer "who labeled this and when" after a model-drift investigation.

The mapping for ``class_id_history`` lives in
``src/clients/curation_opensearch.py``. Writers use the helpers here.

For region-of-interest writes, the ``RegionFields.detector_chain``
keyword-array field already captures per-event provenance; this module
exposes a helper that appends to it consistently.
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

MAX_PLATE_CHAIN_ENTRIES = 16
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
    if len(history) > MAX_HISTORY_ENTRIES and not current_source.get('class_validated'):
        # Drop-oldest, preserving any seed_backfill stub at index 0 if
        # present so we keep the original origin marker. Skipped
        # entirely once class_validated=true.
        if history and history[0].get('writer') == 'seed_backfill':
            history = [history[0], *history[-(MAX_HISTORY_ENTRIES - 1) :]]
        else:
            history = history[-MAX_HISTORY_ENTRIES:]
    return history


def append_plate_chain_entry(
    current_chain: list[str] | None,
    *,
    detector: str,
    detector_version: str,
    outcome: str,
    at: str | None = None,
) -> list[str]:
    """Append one cascade event to a region detector-chain array
    (``RegionFields.detector_chain``).

    Each entry is a colon-delimited string ``{detector}:{version}:{outcome}@{iso}``.
    The cap is :data:`MAX_PLATE_CHAIN_ENTRIES`; drop-oldest on overflow.

    Args:
        current_chain: Existing chain or None.
        detector: Detector identifier, e.g. a primary detector name,
            a secondary detector name, or ``human``.
        detector_version: Detector version string.
        outcome: ``hit``, ``miss``, ``verify_ok``, ``verify_rejected``, ``draw``.
        at: Optional ISO-8601 timestamp; defaults to UTC now.

    Returns:
        The new chain array.
    """
    chain = list(current_chain or [])
    entry = f'{detector}:{detector_version}:{outcome}@{at or _now_iso()}'
    chain.append(entry)
    if len(chain) > MAX_PLATE_CHAIN_ENTRIES:
        chain = chain[-MAX_PLATE_CHAIN_ENTRIES:]
    return chain


__all__ = [
    'MAX_HISTORY_ENTRIES',
    'MAX_PLATE_CHAIN_ENTRIES',
    'append_plate_chain_entry',
    'record_class_history',
]
