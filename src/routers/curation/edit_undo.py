"""Curation router sub-module — undo of the human edits that aren't class
writes: region writes and VLM-suggestion dismissals.

Every human region writer (``PUT /crops/{id}/region``, ``PUT
/crops/batch_region``, ``PATCH /crops/{id}/region_meta``, ``POST
/regions/batch_status``) snapshots the item's pre-write region state into
its edit history (:mod:`src.services.curation.edit_history`). These routes
put that state back — box, score, status, verified/validated flags,
detector/verifier provenance, text and region-cluster placement — the
same way ``POST /crops/{id}/label/undo`` does for class writes.
``POST /crops/{id}/vlm_dismiss`` snapshots the dismissal fields it
replaces the same way, so ``POST /crops/{id}/vlm_dismiss/undo`` brings the
dismissed suggestion back.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.routers.curation._common import OpenSearchDep, RegionProfileDep, _now_iso, logger, router
from src.routers.curation.label_undo import _items_by_ids
from src.services.curation.edit_history import (
    EDIT_HISTORY_FIELD,
    REGION_UNDO_WRITER,
    VLM_DISMISS_UNDO_WRITER,
    EditKind,
    find_edit_undo,
    record_edit,
    restore_edit_state,
)


class CropRegionUndoBatchRequest(BaseModel):
    """Undo the most recent human region write on each crop."""

    model_config = ConfigDict(extra='forbid')

    crop_ids: list[str] = Field(..., min_length=1)


class NothingToUndoError(Exception):
    """The item has no un-undone edit of the requested kind on record."""


def edit_undo_merger(kind: EditKind, writer: str) -> Any:
    """OCC merger restoring the state before the latest un-undone ``kind``
    edit and recording the replaced state (``restorable=False``)."""

    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        entry = find_edit_undo(current.get(EDIT_HISTORY_FIELD), kind)
        if entry is None:
            raise NothingToUndoError
        history = record_edit(current, kind=kind, writer=writer, restorable=False)
        return {
            **restore_edit_state(entry, kind),
            EDIT_HISTORY_FIELD: history,
            'updated_at': _now_iso(),
        }

    return _merge


async def undo_edit(opensearch: Any, crop_id: str, kind: EditKind, writer: str) -> None:
    """Restore one crop. Raises NothingToUndoError, OCCFinalConflictError,
    or HTTPException(404)."""
    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=edit_undo_merger(kind, writer),
            refresh=True,
            writer_id=writer,
        )
    except (NothingToUndoError, OCCFinalConflictError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc


@router.post('/crops/{crop_id}/region/undo')
async def undo_crop_region(
    crop_id: str, opensearch: OpenSearchDep, _profile: RegionProfileDep
) -> dict[str, Any]:
    """Restore the crop's region to its state before its most recent human
    region write (confirm, reject, false positive, box edit, status or text
    change). Repeated calls step back through successive writes. Returns
    the restored item (shared wire format). ``409`` when the crop has no
    human region write left to undo."""
    try:
        await undo_edit(opensearch, crop_id, EditKind.REGION, REGION_UNDO_WRITER)
    except NothingToUndoError as exc:
        raise HTTPException(
            status_code=409, detail=f'nothing to undo for crop region {crop_id}'
        ) from exc
    items = await _items_by_ids(opensearch, [crop_id])
    if not items:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}')
    return items[0]


@router.post('/crops/region/undo_batch')
async def undo_crop_regions(
    payload: CropRegionUndoBatchRequest, opensearch: OpenSearchDep, _profile: RegionProfileDep
) -> dict[str, Any]:
    """Batch form of ``POST /crops/{crop_id}/region/undo`` — undo a bulk
    region write by passing the same ``crop_ids``. Each crop is restored
    independently. Returns ``items`` (restored wire items), ``undone``, and
    per-crop ``nothing_to_undo`` / ``conflicts`` / ``not_found`` id lists.
    ``409`` when no crop had anything to undo."""
    crop_ids = list(dict.fromkeys(payload.crop_ids))
    if not crop_ids:
        return {'items': [], 'undone': 0, 'nothing_to_undo': [], 'conflicts': [], 'not_found': []}

    async def _one(crop_id: str) -> str:
        try:
            await undo_edit(opensearch, crop_id, EditKind.REGION, REGION_UNDO_WRITER)
        except NothingToUndoError:
            return 'nothing_to_undo'
        except OCCFinalConflictError:
            return 'conflicts'
        except HTTPException:
            return 'not_found'
        except Exception as exc:
            logger.warning('undo_region_failed', crop_id=crop_id, error=str(exc))
            return 'conflicts'
        return 'undone'

    outcomes = await asyncio.gather(*(_one(cid) for cid in crop_ids))
    by: dict[str, list[str]] = {
        'undone': [],
        'nothing_to_undo': [],
        'conflicts': [],
        'not_found': [],
    }
    for crop_id, outcome in zip(crop_ids, outcomes, strict=True):
        by[outcome].append(crop_id)
    if not by['undone'] and by['nothing_to_undo'] and not (by['conflicts'] or by['not_found']):
        raise HTTPException(status_code=409, detail='nothing to undo for any crop region')
    return {
        'items': await _items_by_ids(opensearch, by['undone']),
        'undone': len(by['undone']),
        'nothing_to_undo': by['nothing_to_undo'],
        'conflicts': by['conflicts'],
        'not_found': by['not_found'],
    }


@router.post('/crops/{crop_id}/vlm_dismiss/undo')
async def undo_vlm_dismiss(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Undo the most recent ``POST /crops/{crop_id}/vlm_dismiss``: the
    ``vlm_dismissed_*`` fields go back to what they were before it, so the
    dismissed VLM suggestion is live again (``vlm_proposed_class_*`` /
    ``proposed_class_*`` apply it). Returns the restored item; ``409`` when
    there is no dismissal left to undo."""
    try:
        await undo_edit(opensearch, crop_id, EditKind.VLM_DISMISS, VLM_DISMISS_UNDO_WRITER)
    except NothingToUndoError as exc:
        raise HTTPException(
            status_code=409, detail=f'no VLM dismissal to undo for crop {crop_id}'
        ) from exc
    items = await _items_by_ids(opensearch, [crop_id])
    if not items:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}')
    return items[0]
