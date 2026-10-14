"""Curation router sub-module — whole-set human region edit endpoints.

``PATCH /crops/{id}/region_meta`` and ``POST /regions/batch_status``, plus
the shared OCC write plumbing (:class:`_Recorder`, :func:`_write_one`,
:func:`_batch_write`) the per-box routes in
:mod:`src.routers.curation.regions_boxes_edit` reuse. Every write builds
its update document in :mod:`src.services.curation.region_writes` (so the
region lifecycle invariants hold whichever writer a client picks) and
snapshots the pre-write region state into the item's edit history, which
``POST /crops/{id}/region/undo`` restores. Split from
:mod:`src.routers.curation.regions` (browse) so each module holds one
concern.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.clients.occ import OCC_BULK_MGET_SOURCE_EXCLUDES, OCCFinalConflictError, occ_update_one
from src.clients.occ_bulk import BulkFailure, occ_bulk_rounds
from src.config import get_region_fields
from src.domain.region_boxes import RegionBoxWriteError
from src.routers.curation._common import (
    OpenSearchDep,
    RegionProfileDep,
    items_index,
    logger,
    router,
)
from src.routers.curation._common_models import (
    HUMAN_REGION_STATUS_VALUES,
    CropBatchStatusRequest,
    ItemRegionMetaRequest,
)
from src.routers.curation._error_models import REGION_PROFILE_RESPONSES
from src.routers.curation._region_write_models import (
    RegionBatchWriteResponse,
    RegionMetaPatchResponse,
)
from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind, record_edit
from src.services.curation.region_box_refresh import refresh_box_embeddings
from src.services.curation.region_rows import as_row
from src.services.curation.region_writes import (
    human_status_box_write,
    label_source_is_human,
    post_write_item,
    reason_only_box_write,
)
from src.utils.timeutil import now_iso


def _write_error(exc: RegionBoxWriteError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


def _validate_status(region_status: str | None) -> None:
    if region_status not in HUMAN_REGION_STATUS_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f'region_status must be one of {sorted(HUMAN_REGION_STATUS_VALUES)}; '
            f'got {region_status!r}',
        )


class _Recorder:
    """OCC merger wrapper that remembers the doc it last merged onto, so the
    handler can return the post-write item without a second read.

    Every write it builds also snapshots the pre-write region state into
    the item's edit history, which ``POST /crops/{id}/region/undo``
    restores."""

    def __init__(self, build: Any, writer: str) -> None:
        self._build = build
        self._writer = writer
        self.current: dict[str, Any] = {}
        self.update: dict[str, Any] = {}

    def __call__(self, current: dict[str, Any]) -> dict[str, Any]:
        self.current = current
        self.update = {
            **self._build(current),
            EDIT_HISTORY_FIELD: record_edit(current, kind=EditKind.REGION, writer=self._writer),
        }
        return self.update

    def item(self, crop_id: str) -> dict[str, Any]:
        return post_write_item(self.current, self.update, crop_id)


async def _write_one(
    opensearch: Any, crop_id: str, rec: _Recorder, writer_id: str, **kw: Any
) -> None:
    """One OCC write; RegionBoxWriteError -> 422, a missing doc -> 404."""
    try:
        await occ_update_one(
            opensearch, doc_id=crop_id, merger=rec, refresh='wait_for', writer_id=writer_id, **kw
        )
    except OCCFinalConflictError:
        raise
    except RegionBoxWriteError as exc:
        raise _write_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc


@router.patch(
    '/crops/{crop_id}/region_meta',
    response_model=None,
    responses={**REGION_PROFILE_RESPONSES, 200: {'model': RegionMetaPatchResponse}},
)
async def patch_crop_region_meta(
    crop_id: str,
    payload: ItemRegionMetaRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
) -> dict[str, Any]:
    """Patch region metadata (status / rejection reason).

    Box edits (geometry, per-box text) go through ``PUT
    /crops/{crop_id}/regions`` / ``PATCH /crops/{crop_id}/regions/{box_id}``
    -- ``region_text`` is no longer accepted here (D decision,
    2026-09-26). Only the fields present in the payload are written; a
    status write applies the lifecycle invariants (see
    :mod:`src.services.curation.region_writes`). Returns
    ``updated_fields`` (wire names) and ``item`` (post-write).
    """
    F = get_region_fields()
    fields_set = payload.model_fields_set
    if not (fields_set - {'region_label_source'}):
        raise HTTPException(
            status_code=400,
            detail='at least one of region_status, region_rejection_reason must be provided',
        )
    if 'region_status' in fields_set:
        _validate_status(payload.region_status)

    base: dict[str, Any] = {'updated_at': now_iso()}
    # Wire names of the fields this request changed — never `doc.keys()`,
    # which are RegionFields storage keys.
    wire_fields: list[str] = []
    if 'region_status' in fields_set:
        base[F.label_source] = payload.region_label_source
        wire_fields.append('region_status')
    if 'region_rejection_reason' in fields_set:
        wire_fields.append('region_rejection_reason')
    # Operator-initiated edits are terminal — keep the row out of the
    # /review/regions queue. AI-source patches (auto-relabel jobs) skip
    # this so they remain reviewable. Region signal only.
    if label_source_is_human(payload.region_label_source):
        base[F.validated] = True

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        doc = dict(base)
        if 'region_status' in fields_set:
            doc.update(
                human_status_box_write(
                    str(payload.region_status),
                    current,
                    rejection_reason=payload.region_rejection_reason
                    if 'region_rejection_reason' in fields_set
                    else None,
                    label_source=payload.region_label_source,
                )
            )
        elif 'region_rejection_reason' in fields_set:
            # No status change, just editing the rejection
            # reason already on the item's rejected box(es) -- the retired
            # item-level region_rejection_reason scalar covered this same
            # case unconditionally; now it patches every currently-rejected
            # box (or, box-less, the item mirror; see
            # reason_only_box_write's docstring).
            doc.update(reason_only_box_write(current, payload.region_rejection_reason))
        return doc

    rec = _Recorder(_build, 'human:patch_region_meta')
    await _write_one(opensearch, crop_id, rec, 'human:patch_region_meta')
    embedding = await refresh_box_embeddings(opensearch, [crop_id])
    return {
        'crop_id': crop_id,
        'updated_fields': sorted(wire_fields),
        'item': rec.item(crop_id),
        'vector_refresh': embedding,
    }


def empty_batch_result() -> dict[str, Any]:
    """The batch-write response for a batch that names nothing to write."""
    return {
        'updated': 0,
        'conflicts': [],
        'invalid': [],
        'items': [],
        'vector_refresh': {'embedded': 0, 'pending': 0},
    }


async def _batch_write(
    opensearch: Any,
    crop_ids: list[str],
    build: Any,
    writer_id: str,
) -> dict[str, Any]:
    """Apply ``build`` to every crop through :func:`occ_bulk_rounds`.

    A region-batch write has a third outcome besides written and conflicted:
    ``invalid`` (a :class:`RegionBoxWriteError` from ``build``), which must never
    be reported as updated, so it uses the callback form rather than
    :func:`occ_update_bulk`. A bulk request that fails as a whole is reported
    as a conflict, not retried.
    """
    F = get_region_fields()
    conflicts: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    recorders: dict[str, _Recorder] = {}

    def prepare(crop_id: str, doc: dict[str, Any] | None) -> dict[str, Any] | None:
        if doc is None:
            conflicts.append({'crop_id': crop_id, 'current_source': None})
            return None
        rec = _Recorder(build, writer_id)
        try:
            update_doc = rec(doc.get('_source') or {})
        except RegionBoxWriteError as exc:
            invalid.append({'crop_id': crop_id, 'detail': str(exc)})
            return None
        recorders[crop_id] = rec
        return update_doc

    def on_written(crop_id: str) -> None:
        items.append(as_row(recorders[crop_id].item(crop_id), None))

    def on_failed(crop_id: str, doc: dict[str, Any] | None, why: BulkFailure) -> None:
        if why == 'transport':
            logger.warning('batch_region_write_failed', writer_id=writer_id)
            conflicts.append({'crop_id': crop_id, 'current_source': None})
            return
        source = ((doc or {}).get('_source') or {}).get(F.label_source)
        conflicts.append({'crop_id': crop_id, 'current_source': source})

    await occ_bulk_rounds(
        opensearch,
        index=items_index(),
        ids=crop_ids,
        prepare=prepare,
        on_written=on_written,
        on_failed=on_failed,
        max_retries=2,
        refresh='wait_for',
        retry_transport=False,
        source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES,
    )
    return {'updated': len(items), 'conflicts': conflicts, 'invalid': invalid, 'items': items}


@router.post(
    '/regions/batch_status',
    response_model=None,
    responses={**REGION_PROFILE_RESPONSES, 200: {'model': RegionBatchWriteResponse}},
)
async def batch_set_region_status(
    payload: CropBatchStatusRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
) -> dict[str, Any]:
    """Bulk-set region status over many crops — the cluster-view triage op.

    Applies the same lifecycle invariants as ``PATCH region_meta``:
    ``no_region_visible`` clears the box, ``region_verified`` follows the
    status (a request's ``region_verified`` is ignored), ``detected``
    without a box lands in ``invalid``. Human edits are terminal
    (``region_validated=True``). Returns ``updated``, ``conflicts``,
    ``invalid`` and ``items`` (post-write rows: the wire item plus
    ``region_box_id: null``, a whole-set write being item-level).
    """
    F = get_region_fields()
    if not payload.crop_ids:
        return empty_batch_result()
    _validate_status(payload.region_status)
    base: dict[str, Any] = {
        F.label_source: payload.region_label_source,
        'updated_at': now_iso(),
    }
    if label_source_is_human(payload.region_label_source):
        base[F.validated] = True

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        return {
            **base,
            **human_status_box_write(
                payload.region_status, current, label_source=payload.region_label_source
            ),
        }

    result = await _batch_write(
        opensearch, payload.crop_ids, _build, 'human:batch_set_region_status'
    )
    embedding = await refresh_box_embeddings(
        opensearch, [item['crop_id'] for item in result['items']]
    )
    return {**result, 'vector_refresh': embedding}
