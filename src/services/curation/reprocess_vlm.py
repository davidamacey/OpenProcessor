"""The ``vlm`` scope (W10.13): clear a VLM class answer so the VLM worker and
auto-label select the item again.

An unlocked item whose ``class_source`` came from a VLM reply is put back to
the class state recorded by the VLM writer's own pre-write snapshot
(:func:`~src.services.curation.history.restore_class_state`), and its
attempt markers are reset. An item with no VLM snapshot is left alone
(nothing is invented), and a locked class (human, validated import,
holdout) is never touched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import get_curation_config
from src.services.curation.class_sources import VLM_CLASS_SOURCES
from src.services.curation.history import record_class_snapshot, restore_class_state
from src.services.curation.reprocess_locks import class_locked, class_locked_clause
from src.services.curation.vlm_class_attempt import (
    VLM_CLASS_ATTEMPTED_AT_FIELD,
    VLM_CLASS_EMPTY_REASON_FIELD,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

REPROCESS_VLM_WRITER = 'reprocess:vlm'

# Fields only a VLM reply writes; they describe the answer being cleared.
_VLM_ANSWER_FIELDS = (
    VLM_CLASS_ATTEMPTED_AT_FIELD,
    VLM_CLASS_EMPTY_REASON_FIELD,
    'vlm_raw_class',
    'vlm_proposed_class',
    'needs_new_class',
)


def vlm_class_clause() -> dict[str, Any]:
    return {'terms': {'class_source': sorted(VLM_CLASS_SOURCES)}}


def vlm_snapshot(history: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """The newest restorable entry a VLM writer recorded: the class state
    that writer replaced."""
    for entry in reversed(history or []):
        writer = entry.get('writer')
        if isinstance(writer, str) and writer.startswith('vlm') and entry.get('restorable'):
            return entry
    return None


def vlm_restore_update(current: dict[str, Any], *, now: str) -> dict[str, Any] | None:
    """The merge body clearing ``current``'s VLM answer, or ``None`` when
    there is nothing to clear (locked, not a VLM class, no snapshot)."""
    if class_locked(current) or current.get('class_source') not in VLM_CLASS_SOURCES:
        return None
    snapshot = vlm_snapshot(current.get('class_id_history'))
    if snapshot is None:
        return None
    update = restore_class_state(snapshot)
    update.update(dict.fromkeys(_VLM_ANSWER_FIELDS))
    update['class_id_history'] = record_class_snapshot(
        current, writer=REPROCESS_VLM_WRITER, restorable=False, now=now
    )
    update['updated_at'] = now
    return update


async def apply_vlm(opensearch: AsyncOpenSearch, ids: list[str]) -> int:
    """Clear the VLM answer on every id that is still eligible against its
    fresh document; returns how many were rewritten."""
    if not ids:
        return 0
    now = datetime.now(UTC).isoformat()

    def _merge(_doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        return vlm_restore_update(current, now=now) or {}

    cfg = get_curation_config()
    updated = 0
    for start in range(0, len(ids), 500):
        result = await occ_skip_on_conflict_bulk(
            opensearch,
            doc_ids=ids[start : start + 500],
            merger=_merge,
            index=cfg.items_index,
            refresh=False,
            writer_id='reprocess_vlm',
        )
        updated += int(result.get('updated', 0))
    if updated:
        await opensearch.indices.refresh(index=cfg.items_index)
    return updated


__all__ = [
    'REPROCESS_VLM_WRITER',
    'apply_vlm',
    'class_locked_clause',
    'vlm_class_clause',
    'vlm_restore_update',
    'vlm_snapshot',
]
