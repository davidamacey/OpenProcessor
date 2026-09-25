"""Curation router sub-module — human region-of-interest edit endpoints.

``PUT /crops/{id}/region`` (+ the batch form), ``PATCH
/crops/{id}/region_meta`` and ``POST /regions/batch_status``. Every write
builds its update document in :mod:`src.services.curation.region_writes`
(so the region lifecycle invariants hold whichever writer a client picks)
and snapshots the pre-write region state into the item's edit history,
which ``POST /crops/{id}/region/undo`` restores. Split from
:mod:`src.routers.curation.regions` (browse) so each module holds one
concern.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.config import get_region_fields
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    HUMAN_REGION_STATUS_VALUES,
    CropBatchStatusRequest,
    ItemBatchRegionRequest,
    ItemRegionMetaRequest,
    ItemRegionRequest,
    OpenSearchDep,
    RegionProfileDep,
    _now_iso,
    logger,
    router,
)
from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind, record_edit
from src.services.curation.region_writes import (
    RegionWriteError,
    human_status_fields,
    parent_to_source_bbox,
    post_write_item,
    region_box_write,
    validate_bbox_norm,
)
from src.services.curation.wire import region_wire_key
from src.services.detection.region_text import TEXT_CHOICE_HUMAN


def _write_error(exc: RegionWriteError) -> HTTPException:
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
    """One OCC write; RegionWriteError -> 422, a missing doc -> 404."""
    try:
        await occ_update_one(
            opensearch, doc_id=crop_id, merger=rec, refresh='wait_for', writer_id=writer_id, **kw
        )
    except OCCFinalConflictError:
        raise
    except RegionWriteError as exc:
        raise _write_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc


def _box_builder(payload: ItemRegionRequest | ItemBatchRegionRequest) -> Any:
    """Merger body for a box write. Range errors are a 400 up front; a
    parent-frame box is projected per item inside the merger, and a box
    equal to the stored one is a confirmation (detector provenance kept)."""
    box = None if payload.region_bbox_norm is None else list(payload.region_bbox_norm)
    if box is not None:
        try:
            validate_bbox_norm(box)
        except RegionWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    now = _now_iso()
    source = payload.region_label_source

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        target = box
        if box is not None and payload.frame != 'source':
            target = parent_to_source_bbox(box, current.get('bbox_norm'))
        return region_box_write(current, target, label_source=source, now=now)

    return _build


@router.put('/crops/{crop_id}/region')
async def set_crop_region(
    crop_id: str,
    payload: ItemRegionRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
) -> dict[str, Any]:
    """Set or clear the region sub-bbox on a single crop.

    ``region_bbox_norm`` is in ``frame`` (``source`` default, or
    ``parent`` = the item crop, projected server-side). ``None`` clears the box and marks the crop
    ``region_status='no_region_visible'``. A box equal to the stored one
    (within float noise) confirms the region: status, verified, validated
    and verifier are written, the detector / version / score / detection
    time are kept. A different box is human geometry (detector = human,
    score 1.0). Returns ``item``, the post-write wire item.
    """
    F = get_region_fields()
    rec = _Recorder(_box_builder(payload), 'human:set_crop_region')
    await _write_one(opensearch, crop_id, rec, 'human:set_crop_region')
    return {
        'crop_id': crop_id,
        region_wire_key('bbox_norm'): rec.update[F.bbox_norm],
        region_wire_key('status'): rec.update[F.status],
        'item': rec.item(crop_id),
    }


@router.patch('/crops/{crop_id}/region_meta')
async def patch_crop_region_meta(
    crop_id: str,
    payload: ItemRegionMetaRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
) -> dict[str, Any]:
    """Patch region metadata (text / status / rejection reason).

    Bbox edits go through ``PUT /crops/{crop_id}/region``. Only the fields
    present in the payload are written; a status write applies the
    lifecycle invariants (see :mod:`src.services.curation.region_writes`).
    Returns ``updated_fields`` (wire names) and ``item`` (post-write).
    """
    F = get_region_fields()
    fields_set = payload.model_fields_set
    if not (fields_set - {'region_label_source'}):
        raise HTTPException(
            status_code=400,
            detail='at least one of region_text, region_status, '
            'region_rejection_reason must be provided',
        )
    if 'region_status' in fields_set:
        _validate_status(payload.region_status)

    base: dict[str, Any] = {'updated_at': _now_iso()}
    # Wire names of the fields this request changed — never `doc.keys()`,
    # which are RegionFields storage keys.
    wire_fields: list[str] = []
    if 'region_text' in fields_set:
        # Human-typed text is the ground truth; mark the source so the
        # OCR pipeline knows not to overwrite it. Human OCR is 1.0
        # confidence — null would read as "unknown".
        base[F.text] = payload.region_text
        base[F.text_source] = 'human'
        base[F.text_confidence] = 1.0 if payload.region_text else None
        base[F.text_choice] = TEXT_CHOICE_HUMAN
        wire_fields.append('region_text')
    if 'region_status' in fields_set:
        base[F.label_source] = payload.region_label_source
        wire_fields.append('region_status')
    if 'region_rejection_reason' in fields_set:
        base[F.rejection_reason] = payload.region_rejection_reason
        wire_fields.append('region_rejection_reason')
    # Operator-initiated edits are terminal — keep the row out of the
    # /review/regions queue. AI-source patches (auto-relabel jobs) skip
    # this so they remain reviewable. Region signal only.
    if (payload.region_label_source or '').lower().startswith('human'):
        base[F.validated] = True

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        doc = dict(base)
        if 'region_status' in fields_set:
            doc.update(human_status_fields(str(payload.region_status), current))
        return doc

    rec = _Recorder(_build, 'human:patch_region_meta')
    await _write_one(opensearch, crop_id, rec, 'human:patch_region_meta')
    return {'crop_id': crop_id, 'updated_fields': sorted(wire_fields), 'item': rec.item(crop_id)}


async def _batch_write(
    opensearch: Any,
    crop_ids: list[str],
    build: Any,
    writer_id: str,
) -> dict[str, Any]:
    """Apply ``build`` to every crop via one batched mget + bulk round-trip
    per retry round, instead of one ``occ_update_one`` round-trip
    per crop.

    Doesn't route through :func:`src.clients.occ_bulk.occ_update_bulk` — that
    helper's merge_fn contract only distinguishes "wrote" vs "nothing to
    write" (a falsy return is a documented noop counted as updated), but
    a region-batch write needs a third outcome (``invalid``, a
    :class:`RegionWriteError` from ``build``) that must never be reported
    as updated. ``refresh`` is attached only to the final bulk call of
    the final retry round — no forced ``indices.refresh`` per call.
    """
    from src.clients.curation_opensearch import mget_crops
    from src.clients.occ import OCC_BULK_MGET_SOURCE_EXCLUDES, OCC_BULK_PAGE_SIZE

    F = get_region_fields()
    updated = 0
    conflicts: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    pending_ids = list(dict.fromkeys(crop_ids))
    max_retries = 2
    for attempt in range(max_retries + 1):
        if not pending_ids:
            break
        next_round: list[str] = []
        page_starts = list(range(0, len(pending_ids), OCC_BULK_PAGE_SIZE))
        for page_idx, start in enumerate(page_starts):
            page_ids = pending_ids[start : start + OCC_BULK_PAGE_SIZE]
            docs = await mget_crops(
                opensearch,
                page_ids,
                index=CURATION_ITEMS_INDEX,
                source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES,
                seq_no=True,
            )

            pending: list[tuple[str, dict[str, Any], _Recorder]] = []
            for crop_id in page_ids:
                doc = docs.get(crop_id)
                if doc is None:
                    conflicts.append({'crop_id': crop_id, 'current_source': None})
                    continue
                source = doc.get('_source') or {}
                rec = _Recorder(build, writer_id)
                try:
                    update_doc = rec(source)
                except RegionWriteError as exc:
                    invalid.append({'crop_id': crop_id, 'detail': str(exc)})
                    continue
                pending.append((crop_id, update_doc, rec))

            if not pending:
                continue

            bulk_body: list[dict[str, Any]] = []
            for crop_id, update_doc, _rec in pending:
                doc = docs[crop_id]
                bulk_body.append(
                    {
                        'update': {
                            '_index': CURATION_ITEMS_INDEX,
                            '_id': crop_id,
                            'if_seq_no': doc['_seq_no'],
                            'if_primary_term': doc['_primary_term'],
                        }
                    }
                )
                bulk_body.append({'doc': update_doc})

            is_last_page = page_idx == len(page_starts) - 1
            call_refresh: bool | str = 'wait_for' if is_last_page else False
            try:
                resp = await opensearch.bulk(body=bulk_body, refresh=call_refresh)
            except Exception as exc:
                logger.warning('batch_region_write_failed', writer_id=writer_id, error=str(exc))
                for crop_id, _update_doc, _rec in pending:
                    conflicts.append({'crop_id': crop_id, 'current_source': None})
                continue

            resp_items = resp.get('items') or []
            for (crop_id, _update_doc, rec), item in zip(pending, resp_items, strict=True):
                action = item.get('update') or {}
                status = action.get('status')
                if status in (200, 201):
                    updated += 1
                    items.append(rec.item(crop_id))
                    continue
                error = action.get('error') or {}
                is_conflict = status == 409 or 'version_conflict' in error.get('type', '')
                if is_conflict and attempt < max_retries:
                    next_round.append(crop_id)
                else:
                    current_source = docs.get(crop_id, {}).get('_source', {}).get(F.label_source)
                    conflicts.append({'crop_id': crop_id, 'current_source': current_source})

        pending_ids = next_round

    return {'updated': updated, 'conflicts': conflicts, 'invalid': invalid, 'items': items}


@router.put('/crops/batch_region')
async def batch_set_crop_region(
    payload: ItemBatchRegionRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
) -> dict[str, Any]:
    """Bulk variant of ``PUT /crops/{crop_id}/region`` (typically
    ``region_bbox_norm=null``: "no region visible on these N crops").

    Returns ``updated``, ``conflicts``, ``invalid`` and ``items`` (the
    post-write wire items of the updated crops).
    """
    if not payload.crop_ids:
        return {'updated': 0, 'conflicts': [], 'invalid': [], 'items': []}
    return await _batch_write(
        opensearch, payload.crop_ids, _box_builder(payload), 'human:batch_set_crop_region'
    )


@router.post('/regions/batch_status')
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
    ``invalid`` and ``items`` (post-write wire items).
    """
    F = get_region_fields()
    if not payload.crop_ids:
        return {'updated': 0, 'conflicts': [], 'invalid': [], 'items': []}
    _validate_status(payload.region_status)
    base: dict[str, Any] = {
        F.label_source: payload.region_label_source,
        'updated_at': _now_iso(),
    }
    if (payload.region_label_source or '').lower().startswith('human'):
        base[F.validated] = True

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        return {**base, **human_status_fields(payload.region_status, current)}

    return await _batch_write(opensearch, payload.crop_ids, _build, 'human:batch_set_region_status')
