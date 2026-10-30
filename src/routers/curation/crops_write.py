"""Curation router sub-module — item (crop) label, move and flag-new-class writes."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.routers.curation._common import (
    OpenSearchDep,
    RegistryDep,
    get_class_registry,
    items_index,
    router,
)
from src.routers.curation._common_models import (
    CropBatchLabelRequest,
    CropFlagNewClassRequest,
    CropLabelRequest,
    CropMoveRequest,
)
from src.routers.curation._selection import selected_crop_ids
from src.routers.curation._selection_write_models import (
    BatchRelabelResponse,
    SelectionDryRunResponse,
)
from src.services.curation.class_label import (
    ItemLabel,
    candidate_move_update,
    class_label_update,
    human_move_class_update,
)
from src.services.curation.cluster_ids import cluster_kind
from src.services.curation.label_ops.record import recorded_relabel
from src.services.curation.label_ops.store import NewLabelOp
from src.utils.timeutil import now_iso


@router.put('/crops/{crop_id}/label')
async def label_crop(
    crop_id: str,
    payload: CropLabelRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> dict[str, Any]:
    """Set the validated class on a single item."""
    reg = registry
    if not reg.validate_id(payload.class_id):
        raise HTTPException(status_code=400, detail=f'unknown class_id {payload.class_id}')
    entry = reg.get(payload.class_id)
    class_name = entry.class_name if entry is not None else ''

    def _merge_label(current: dict[str, Any]) -> dict[str, Any]:
        return class_label_update(
            current,
            ItemLabel.human(
                class_id=payload.class_id,
                class_name=class_name,
                label_source=payload.label_source,
                writer='human:label_crop',
            ),
        )

    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=_merge_label,
            refresh='wait_for',
            writer_id='human:label_crop',
        )
    except OCCFinalConflictError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    return {'crop_id': crop_id, 'class_id': payload.class_id, 'class_name': class_name}


@router.put(
    '/crops/batch_label',
    response_model=None,
    responses={200: {'model': BatchRelabelResponse | SelectionDryRunResponse}},
)
async def batch_label_crops(
    payload: CropBatchLabelRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Bulk label many crops with a single class_id.

    Each crop is updated via ``occ_update_one`` so a concurrent worker
    write can't silently clobber the human edit and the prior class
    assignment is snapshotted into ``class_id_history``. Returns
    ``{updated, conflicts: [{crop_id, current_source}]}`` — conflicts
    list crops whose OCC retries were exhausted.
    """
    reg = get_class_registry()
    if not reg.validate_id(payload.class_id):
        raise HTTPException(status_code=400, detail=f'unknown class_id {payload.class_id}')
    entry = reg.get(payload.class_id)
    class_name = entry.class_name if entry is not None else ''

    ids = await selected_crop_ids(opensearch, payload)
    if payload.dry_run:
        return {'dry_run': True, 'selected': len(ids)}
    if not ids:
        return {'updated': 0, 'updated_ids': [], 'conflicts': [], 'op_id': None}

    def _merger_for(op_id: str) -> Any:
        def _merge(current: dict[str, Any]) -> dict[str, Any]:
            return class_label_update(
                current,
                ItemLabel.human(
                    class_id=payload.class_id,
                    class_name=class_name,
                    label_source=payload.label_source,
                    writer='human:batch_label_crops',
                    op_id=op_id,
                ),
            )

        return _merge

    # One mget page + one bulk call per page of ids, recorded as an operation
    # so it can be undone by id.
    return await recorded_relabel(
        opensearch,
        ids,
        op=NewLabelOp(
            kind='label',
            writer='human:batch_label_crops',
            class_name=class_name,
            class_id=payload.class_id,
            label_source=payload.label_source,
            target='crop_ids' if payload.crop_ids is not None else 'selection',
            selected=len(ids),
        ),
        merger_for=_merger_for,
    )


@router.post(
    '/crops/move',
    response_model=None,
    responses={200: {'model': BatchRelabelResponse | SelectionDryRunResponse}},
)
async def move_crops(
    payload: CropMoveRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Move crops to a different cluster.

    * Class cluster (``cluster_id == class_id``): also a relabel — the
      crops get that registry class (``class_source='human_move'``,
      ``class_validated=True``): "move this to the X cluster" means "this
      is an X".
    * Candidate cluster: placement only. The crops join the group; a
      human-owned class is cleared (the human just said it isn't that
      class), a machine suggestion is kept, and nothing is validated —
      the group has no class until one is assigned.
    * ``400`` for an unassigned (negative) target (use exclude/discard)
      a class-range id that isn't in the registry, or a candidate id with
      no current members (it doesn't exist). Nothing is written.
    """
    target_id = int(payload.cluster_id)
    kind = cluster_kind(target_id)
    target = None
    if kind == 'unassigned':
        raise HTTPException(
            status_code=400,
            detail=f'cannot move into unassigned cluster {target_id}; use exclude or discard',
        )
    if kind == 'class':
        target = get_class_registry().get(target_id)
        if target is None:
            raise HTTPException(status_code=400, detail=f'unknown class_id {target_id}')
    if kind == 'candidate':
        from src.services.curation.exclusion import cluster_member_count

        if await cluster_member_count(opensearch, items_index(), target_id) == 0:
            raise HTTPException(
                status_code=400, detail=f'candidate cluster {target_id} has no members'
            )
    ids = await selected_crop_ids(opensearch, payload)
    if payload.dry_run:
        return {'dry_run': True, 'selected': len(ids)}
    if not ids:
        return {'updated': 0, 'updated_ids': [], 'conflicts': [], 'op_id': None}

    target_name = target.class_name if target is not None else ''

    def _merger_for(op_id: str) -> Any:
        def _merge(current: dict[str, Any]) -> dict[str, Any]:
            if kind == 'candidate':
                return candidate_move_update(
                    current, cluster_id=target_id, now=now_iso(), op_id=op_id
                )
            # Move-from-cluster is a class gesture ("move this to the X
            # cluster" means "this is an X") — routed through class_label.py,
            # the single class-label writer (tests/test_class_label_single_writer.py).
            return human_move_class_update(
                current, class_id=target_id, class_name=target_name, now=now_iso(), op_id=op_id
            )

        return _merge

    return await recorded_relabel(
        opensearch,
        ids,
        op=NewLabelOp(
            kind='move',
            writer='human:move_crops',
            class_name=target_name if kind == 'class' else None,
            class_id=target_id if kind == 'class' else None,
            target_cluster_id=target_id,
            target_kind='class' if kind == 'class' else 'candidate',
            target='crop_ids' if payload.crop_ids is not None else 'selection',
            selected=len(ids),
        ),
        merger_for=_merger_for,
    )


@router.post('/crops/flag_new_class')
async def flag_new_class(
    payload: CropFlagNewClassRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Flag crops as needing a new class label (curator queue).

    Source imagery can contain object types beyond the current class
    registry. Labelers flag a crop; a curator later batch-reviews the
    queue, decides if a new class warrants creation via
    ``POST /curation/classes``, and labels the flagged crops.
    """
    if not payload.crop_ids:
        return {'updated': 0}
    now = now_iso()
    bulk: list[dict[str, Any]] = []
    for crop_id in payload.crop_ids:
        bulk.append({'update': {'_index': items_index(), '_id': crop_id}})
        bulk.append(
            {
                'doc': {
                    'needs_new_class': True,
                    'needs_new_class_note': payload.note,
                    'needs_new_class_at': now,
                    'updated_at': now,
                }
            }
        )
    try:
        resp = await opensearch.bulk(body=bulk, refresh=False)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    n_errors = sum(1 for it in resp.get('items', []) if any('error' in v for v in it.values()))
    return {'flagged': len(payload.crop_ids) - n_errors, 'errors': n_errors}
