"""Curation router sub-module — undo of the human edits that aren't class
writes: region writes and VLM-suggestion dismissals.

Every human region writer (``PUT /crops/{id}/regions``, ``PUT
/crops/batch_regions``, ``PATCH /crops/{id}/regions/{box_id}``, ``POST
/regions/batch_box_state``, ``PATCH /crops/{id}/region_meta``, ``POST
/regions/batch_status``) snapshots the item's pre-write region state into
its edit history (:mod:`src.services.curation.edit_history`). These routes
put that state back — the whole box list, status, verified/validated
flags and detector/verifier provenance — the same way ``POST
/crops/{id}/label/undo`` does for class writes.
``POST /crops/{id}/vlm_dismiss`` snapshots the dismissal fields it
replaces the same way, so ``POST /crops/{id}/vlm_dismiss/undo`` brings the
dismissed suggestion back.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.config.batch_sync import batch_sync_max
from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._batch_job_models import BatchJobAccepted
from src.routers.curation._common import OpenSearchDep, RegionProfileDep, is_not_found, router
from src.routers.curation._error_models import REGION_PROFILE_RESPONSES
from src.routers.curation.label_undo import _items_by_ids
from src.schemas.curation.label_ops import CropUndoBatchResponse
from src.services.curation.edit_history import REGION_UNDO_WRITER, VLM_DISMISS_UNDO_WRITER, EditKind
from src.services.curation.edit_undo import edit_undo_update, queue_edit_undo_job, undo_edit_writes


UNDO_BATCH_MAX_IDS = 5000


class CropRegionUndoBatchRequest(BaseModel):
    """Undo the most recent human region write on each crop."""

    model_config = ConfigDict(extra='forbid')

    crop_ids: list[str] = Field(..., min_length=1, max_length=UNDO_BATCH_MAX_IDS)


class NothingToUndoError(Exception):
    """The item has no un-undone edit of the requested kind on record."""


def edit_undo_merger(kind: EditKind, writer: str) -> Any:
    """OCC merger restoring the state before the latest un-undone ``kind``
    edit and recording the replaced state (``restorable=False``)."""

    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        update = edit_undo_update(current, kind, writer)
        if update is None:
            raise NothingToUndoError
        return update

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
        if not is_not_found(exc):
            raise
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc


@router.post('/crops/{crop_id}/region/undo', responses=REGION_PROFILE_RESPONSES)
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


@router.post(
    '/crops/region/undo_batch',
    response_model=None,
    responses={
        **REGION_PROFILE_RESPONSES,
        200: {'model': CropUndoBatchResponse},
        202: {'model': BatchJobAccepted},
    },
)
async def undo_crop_regions(
    payload: CropRegionUndoBatchRequest,
    opensearch: OpenSearchDep,
    session: DbSessionDep,
    _profile: RegionProfileDep,
) -> CropUndoBatchResponse | JSONResponse:
    """Batch form of ``POST /crops/{crop_id}/region/undo`` — undo a bulk
    region write by passing the same ``crop_ids``. Each crop is restored
    independently, through the paged bulk OCC loop (one read and one
    conditional write per 500 ids). Returns ``items`` (restored wire items),
    ``undone``, and per-crop ``nothing_to_undo`` / ``conflicts`` /
    ``not_found`` id lists. ``409`` when no crop had anything to undo.

    More than ``OP_BATCH_SYNC_MAX`` crops (default 500) are queued instead:
    ``202`` with ``job_id``; the job's ``result`` at ``GET /jobs/{job_id}``
    carries the same id lists without ``items``."""
    crop_ids = list(dict.fromkeys(payload.crop_ids))
    if len(crop_ids) > batch_sync_max():
        job = await queue_edit_undo_job(session, current_project_id(), crop_ids)
        accepted = BatchJobAccepted(job_id=str(job.id), status=job.status)
        return JSONResponse(status_code=202, content=accepted.model_dump())
    by = await undo_edit_writes(opensearch, crop_ids, EditKind.REGION, REGION_UNDO_WRITER)
    if not by['undone'] and by['nothing_to_undo'] and not (by['conflicts'] or by['not_found']):
        raise HTTPException(status_code=409, detail='nothing to undo for any crop region')
    return CropUndoBatchResponse(
        items=await _items_by_ids(opensearch, by['undone']),
        undone=len(by['undone']),
        nothing_to_undo=by['nothing_to_undo'],
        conflicts=by['conflicts'],
        not_found=by['not_found'],
    )


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
