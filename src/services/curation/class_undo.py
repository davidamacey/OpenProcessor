"""The one restore of a human class write, shared by the per-item undo routes and
the label-operation undo so both write byte-identical updates for the same item."""

from __future__ import annotations

from typing import Any

from src.clients.occ_bulk import occ_bulk_rounds
from src.config.curation_accessors import items_index
from src.domain.audit_math import AUDIT_SAMPLE
from src.domain.history import (
    AUDIT_VERDICT_FIELDS,
    CLASS_STATE_FIELDS,
    HUMAN_UNLABEL_WRITER,
    REVIEW_DISMISS_FIELDS,
    find_undo_snapshot,
    record_class_snapshot,
    restore_class_state,
)
from src.services.curation.exclusion import park_restored_state_while_excluded
from src.utils.timeutil import now_iso


UNDO_READ_FIELDS: list[str] = [
    'crop_id',
    'label_op_id',
    'class_id_history',
    'class_excluded',
    'excluded_at',
    'test_holdout',
    AUDIT_SAMPLE,
    *CLASS_STATE_FIELDS,
    *REVIEW_DISMISS_FIELDS,
    *AUDIT_VERDICT_FIELDS,
]
"""What :func:`class_undo_update` and the operation undo's decision read: never
embeddings or boxes, so a page of 500 documents stays small."""


def class_undo_update(current: dict[str, Any]) -> dict[str, Any]:
    """Merge body putting ``current`` back to its state before its newest un-undone
    human class write. The unlabel entry records which operation it cancelled."""
    snap = find_undo_snapshot(current.get('class_id_history'))
    restored = restore_class_state(snap)
    if current.get('class_excluded'):
        # Still excluded: the restored validation/placement is what
        # un-exclude should bring back, not what applies right now.
        restored = park_restored_state_while_excluded(restored)
    history = record_class_snapshot(
        current,
        writer=HUMAN_UNLABEL_WRITER,
        restorable=False,
        op_id=(snap or {}).get('op_id'),
    )
    return {**restored, 'class_id_history': history, 'updated_at': now_iso()}


async def undo_class_writes(opensearch: Any, crop_ids: list[str]) -> dict[str, list[str]]:
    """Undo the newest human class write of each item, in pages of one conditional bulk.

    Returns the ids per outcome: ``undone``, ``nothing_to_undo`` (no human write left on
    record), ``conflicts`` (a concurrent write outlasted the retries) and ``not_found``.
    """
    outcome: dict[str, str] = {}

    def prepare(doc_id: str, doc: dict[str, Any] | None) -> dict[str, Any] | None:
        if doc is None:
            outcome[doc_id] = 'not_found'
            return None
        source = doc.get('_source') or {}
        if find_undo_snapshot(source.get('class_id_history')) is None:
            outcome[doc_id] = 'nothing_to_undo'
            return None
        outcome[doc_id] = 'conflicts'  # until the bulk reports it written
        return class_undo_update(source)

    def on_written(doc_id: str) -> None:
        outcome[doc_id] = 'undone'

    def on_failed(doc_id: str, _doc: dict[str, Any] | None, _why: str) -> None:
        outcome[doc_id] = 'conflicts'

    await occ_bulk_rounds(
        opensearch,
        index=items_index(),
        ids=crop_ids,
        prepare=prepare,
        on_written=on_written,
        on_failed=on_failed,
        max_retries=5,
        refresh='wait_for',
        source_includes=UNDO_READ_FIELDS,
    )
    by: dict[str, list[str]] = {
        'undone': [],
        'nothing_to_undo': [],
        'conflicts': [],
        'not_found': [],
    }
    for crop_id in crop_ids:
        by[outcome.get(crop_id, 'conflicts')].append(crop_id)
    return by


__all__ = ['UNDO_READ_FIELDS', 'class_undo_update', 'undo_class_writes']
