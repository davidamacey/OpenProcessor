"""The one restore of a human class write, shared by the per-item undo routes and
the label-operation undo so both write byte-identical updates for the same item."""

from __future__ import annotations

from typing import Any

from src.domain.history import (
    HUMAN_UNLABEL_WRITER,
    find_undo_snapshot,
    record_class_snapshot,
    restore_class_state,
)
from src.services.curation.exclusion import park_restored_state_while_excluded
from src.utils.timeutil import now_iso


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


__all__ = ['class_undo_update']
