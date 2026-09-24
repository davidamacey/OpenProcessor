"""Curation router sub-module — undo of human class writes.

Every human class write (``PUT /crops/{id}/label``, ``PUT
/crops/batch_label``, ``POST /crops/move``) records the item's full
pre-write class state in ``class_id_history`` (see
:func:`src.services.curation.history.record_class_snapshot`). The routes
here restore that state, so the labeler never has to decide itself
whether "undo" means re-applying an earlier label or reverting to the
machine suggestion — the backend knows what the item looked like.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    CropUndoBatchRequest,
    OpenSearchDep,
    _now_iso,
    logger,
    router,
)
from src.services.curation.exclusion import park_restored_state_while_excluded
from src.services.curation.history import (
    HUMAN_UNLABEL_WRITER,
    find_undo_snapshot,
    record_class_snapshot,
    restore_class_state,
)
from src.services.curation.wire import item_source_excludes, serialize_item


class NothingToUndoError(Exception):
    """The item has no un-undone human class write on record."""


def _undo_merger(*, require_history: bool) -> Any:
    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        snapshot = find_undo_snapshot(current.get('class_id_history'))
        if snapshot is None and require_history:
            raise NothingToUndoError
        restored = restore_class_state(snapshot)
        if current.get('class_excluded'):
            # Still excluded: the restored validation/placement is what
            # un-exclude should bring back, not what applies right now.
            restored = park_restored_state_while_excluded(restored)
        history = record_class_snapshot(current, writer=HUMAN_UNLABEL_WRITER, restorable=False)
        return {**restored, 'class_id_history': history, 'updated_at': _now_iso()}

    return _merge


async def _undo_one(opensearch: Any, crop_id: str, *, require_history: bool) -> None:
    """Restore one crop. Raises NothingToUndoError, OCCFinalConflictError,
    or HTTPException(404)."""
    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=_undo_merger(require_history=require_history),
            refresh=True,
            writer_id=HUMAN_UNLABEL_WRITER,
        )
    except (NothingToUndoError, OCCFinalConflictError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc


async def _items_by_ids(opensearch: Any, ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    resp = await opensearch.mget(
        index=CURATION_ITEMS_INDEX,
        body={'ids': ids},
        _source_excludes=item_source_excludes(),
    )
    return [
        serialize_item(d.get('_source') or {}, d.get('_id', ''))
        for d in (resp.get('docs') or [])
        if d.get('found')
    ]


@router.post('/crops/{crop_id}/label/undo')
async def undo_crop_label(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Restore the crop to its state before its most recent human class write.

    Whatever that state was — an earlier validated human label, a VLM
    suggestion, an ingest proposal, unlabeled — class, provenance,
    validation and cluster placement come back exactly as recorded.
    Repeated calls step back through successive human writes. Returns the
    restored item (shared wire format). ``409`` when the crop has no human
    class write left to undo.
    """
    try:
        await _undo_one(opensearch, crop_id, require_history=True)
    except NothingToUndoError as exc:
        raise HTTPException(status_code=409, detail=f'nothing to undo for crop {crop_id}') from exc
    items = await _items_by_ids(opensearch, [crop_id])
    if not items:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}')
    return items[0]


@router.post('/crops/label/undo_batch')
async def undo_crop_labels(
    payload: CropUndoBatchRequest, opensearch: OpenSearchDep
) -> dict[str, Any]:
    """Batch form of ``POST /crops/{crop_id}/label/undo``.

    Each crop is restored independently to its own state before its most
    recent human class write — so undoing a ``batch_label`` or ``move``
    means passing the same ``crop_ids``. Returns ``items`` (restored wire
    items), ``undone``, and per-crop ``nothing_to_undo`` / ``conflicts``
    / ``not_found`` id lists. ``409`` when no crop had anything to undo.
    """
    crop_ids = list(dict.fromkeys(payload.crop_ids))
    if not crop_ids:
        return {'items': [], 'undone': 0, 'nothing_to_undo': [], 'conflicts': [], 'not_found': []}

    async def _one(crop_id: str) -> str:
        try:
            await _undo_one(opensearch, crop_id, require_history=True)
        except NothingToUndoError:
            return 'nothing_to_undo'
        except OCCFinalConflictError:
            return 'conflicts'
        except HTTPException:
            return 'not_found'
        except Exception as exc:
            logger.warning('undo_label_failed', crop_id=crop_id, error=str(exc))
            return 'conflicts'
        return 'undone'

    outcomes = await asyncio.gather(*(_one(cid) for cid in crop_ids))
    by_outcome: dict[str, list[str]] = {
        'undone': [],
        'nothing_to_undo': [],
        'conflicts': [],
        'not_found': [],
    }
    for crop_id, outcome in zip(crop_ids, outcomes, strict=True):
        by_outcome[outcome].append(crop_id)
    if (
        not by_outcome['undone']
        and by_outcome['nothing_to_undo']
        and not (by_outcome['conflicts'] or by_outcome['not_found'])
    ):
        raise HTTPException(status_code=409, detail='nothing to undo for any crop')
    return {
        'items': await _items_by_ids(opensearch, by_outcome['undone']),
        'undone': len(by_outcome['undone']),
        'nothing_to_undo': by_outcome['nothing_to_undo'],
        'conflicts': by_outcome['conflicts'],
        'not_found': by_outcome['not_found'],
    }


@router.delete('/crops/{crop_id}/label')
async def unlabel_crop(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Undo the most recent human class label (legacy Undo route).

    Same restore as ``POST /crops/{crop_id}/label/undo``; the one
    difference is that with no human write on record it resets the crop
    to unlabeled (class and provenance cleared, nothing invented) instead
    of answering ``409``. Only the class side is touched; the region-side
    validated flag is independent.
    """
    await _undo_one(opensearch, crop_id, require_history=False)
    return {'crop_id': crop_id, 'reset': True}
