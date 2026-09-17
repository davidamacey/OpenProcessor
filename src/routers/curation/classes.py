"""Curation router sub-module — class registry CRUD + merge."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, status

from src.clients.curation_opensearch import ClassRegistryError
from src.config.region_fields import get_region_fields
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    CURATION_LABELS_CONFIRMED_INDEX,
    ClassCreateRequest,
    ClassEntry,
    ClassListResponse,
    ClassMergeRequest,
    ClassUpdateRequest,
    CropsPageResponse,
    OpenSearchDep,
    _now_iso,
    get_class_registry,
    logger,
    router,
)
from src.routers.curation.crops import list_crops


# Single-char keys the labeler UI's global keydown listener binds to
# labeling actions (accept-vlm, skip, discard, undo, ignore, undo-ignore,
# select-all, move). Binding a class hotkey to one of these creates the
# exact "class letter also fires a global action" collision documented in
# Label Studio #491/#7431 — the class would be assigned *and* the action
# would fire on the same keypress, since both window keydown listeners run
# unconditionally.
RESERVED_HOTKEY_LETTERS = frozenset('gndzxuam')


@router.get('/classes', response_model=ClassListResponse)
async def list_classes(opensearch: OpenSearchDep) -> ClassListResponse:
    """List every class in the registry, with live counts from the items index."""
    reg = get_class_registry().load()
    # Three counts per class, aggregated in one round-trip:
    #   * counts[cid]      = items whose class_id == cid (labeled)
    #   * validated[cid]   = items with class_id == cid AND class_validated=true
    #   * cluster_size[cid]= items whose cluster_id == cid (FAISS bucket, may
    #                        include unlabeled candidates the operator hasn't
    #                        triaged yet). Operators expect the sidebar chip
    #                        to match what they'll see when they click in,
    #                        which is the cluster bucket — not just labeled
    #                        items.
    counts: dict[int, int] = {}
    validated: dict[int, int] = {}
    cluster_size: dict[int, int] = {}
    try:
        body = {
            'size': 0,
            'aggs': {
                'by_class': {
                    'terms': {'field': 'class_id', 'size': 1000},
                    'aggs': {
                        'validated': {
                            'filter': {'term': {'class_validated': True}},
                        },
                    },
                },
                'by_cluster': {
                    'terms': {'field': 'cluster_id', 'size': 1000},
                },
            },
        }
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
        aggs = resp.get('aggregations') or {}
        for bucket in aggs.get('by_class', {}).get('buckets', []):
            cid = int(bucket['key'])
            counts[cid] = int(bucket.get('doc_count', 0))
            validated[cid] = int((bucket.get('validated') or {}).get('doc_count', 0))
        for bucket in aggs.get('by_cluster', {}).get('buckets', []):
            cid = int(bucket['key'])
            cluster_size[cid] = int(bucket.get('doc_count', 0))
    except Exception as exc:
        logger.debug('class_counts_skipped', error=str(exc))

    # A region-of-interest class (e.g. license_plate) lives as a sub-bbox on
    # every parent item that has one, NOT as a separate doc whose primary
    # class_id == that class. The class-aggregation count above only
    # captures the rare mis-labels. Override with the real region
    # inventory, keyed by the configured region-bbox field, so the sidebar
    # matches the region-browse view.
    fields = get_region_fields()
    try:
        region_count = await opensearch.count(
            index=CURATION_ITEMS_INDEX,
            body={'query': {'exists': {'field': fields.bbox_norm}}},
        )
        region_validated = await opensearch.count(
            index=CURATION_ITEMS_INDEX,
            body={
                'query': {
                    'bool': {
                        'must': [
                            {'exists': {'field': fields.bbox_norm}},
                            {'term': {'class_validated': True}},
                        ]
                    }
                }
            },
        )
        for c in reg.classes:
            if (c.class_name or '').lower() == 'license_plate':
                # All three counts share the region inventory total: the
                # region lives as a sub-bbox, not as its own cluster, so
                # there's no separate FAISS bucket to count.
                region_total = int(region_count.get('count', 0))
                region_val = int(region_validated.get('count', 0))
                counts[c.class_id] = region_total
                validated[c.class_id] = region_val
                cluster_size[c.class_id] = region_total
                break
    except Exception as exc:
        logger.debug('region_class_count_skipped', error=str(exc))

    return ClassListResponse(
        classes=[
            ClassEntry(
                class_id=c.class_id,
                class_name=c.class_name,
                group=c.group,
                sample_count=counts.get(c.class_id, c.sample_count),
                validated_count=validated.get(c.class_id, c.validated_count),
                cluster_size=cluster_size.get(c.class_id, 0),
                deprecated=c.deprecated,
                hotkey_letter=getattr(c, 'hotkey_letter', None),
            )
            for c in reg.classes
        ]
    )


@router.post('/classes', status_code=status.HTTP_201_CREATED)
async def create_class(payload: ClassCreateRequest) -> dict[str, Any]:
    """Append-only add."""
    reg = get_class_registry()
    try:
        new_id = reg.add_class(payload.name, group=payload.group, notes=payload.notes)
    except ClassRegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {'class_id': new_id, 'class_name': payload.name, 'group': payload.group}


@router.put('/classes/{class_id}')
async def update_class(class_id: int, payload: ClassUpdateRequest) -> dict[str, Any]:
    """Rename, regroup, or assign/clear a hotkey on a class."""
    reg = get_class_registry()
    try:
        if payload.name is not None:
            reg.rename_class(class_id, payload.name)
        # group + hotkey_letter share the load+mutate+write path.
        if payload.group is not None or payload.hotkey_letter is not None:
            registry_obj = reg.load()
            new_letter: str | None = None
            if payload.hotkey_letter is not None:
                stripped = payload.hotkey_letter.strip()
                # Empty string means "clear the binding".
                if stripped == '':
                    new_letter = None
                else:
                    if len(stripped) != 1:
                        raise HTTPException(
                            status_code=400,
                            detail='hotkey_letter must be a single character',
                        )
                    new_letter = stripped.lower()
                    if new_letter in RESERVED_HOTKEY_LETTERS:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"hotkey '{new_letter}' is reserved for a labeling "
                                'action and cannot be bound to a class'
                            ),
                        )
                    # Uniqueness check across active classes.
                    for c in registry_obj.classes:
                        if c.deprecated or c.class_id == class_id:
                            continue
                        if (getattr(c, 'hotkey_letter', None) or '').lower() == new_letter:
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    f"hotkey '{new_letter}' is already bound to "
                                    f"'{c.class_name}' (class_id={c.class_id})"
                                ),
                            )
            for c in registry_obj.classes:
                if c.class_id == class_id:
                    if payload.group is not None:
                        c.group = payload.group
                    if payload.hotkey_letter is not None:
                        c.hotkey_letter = new_letter
                    break
            reg._atomic_write(registry_obj)
    except HTTPException:
        raise
    except ClassRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    entry = reg.get(class_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f'class_id {class_id} not found')
    return entry.model_dump()


@router.post('/classes/merge')
async def merge_class(payload: ClassMergeRequest, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Mark source deprecated; bulk-relabel matching crops + labels."""
    # A merge that touches frozen test_holdout crops would relabel their
    # class_id and leave their recorded holdout identity (SHA1-of-crop_id-
    # per-class_id) stale. Refuse with 409 naming the affected count rather
    # than silently remapping. Checked before any registry mutation below.
    holdout_count_resp = await opensearch.count(
        index=CURATION_ITEMS_INDEX,
        body={
            'query': {
                'bool': {
                    'must': [{'term': {'class_id': payload.source_id}}],
                    'filter': [{'term': {'test_holdout': True}}],
                },
            },
        },
    )
    holdout_count = int(holdout_count_resp.get('count', 0))
    if holdout_count:
        raise HTTPException(
            status_code=409,
            detail=(
                f'cannot merge class {payload.source_id}: {holdout_count} '
                'frozen test_holdout crop(s) would be relabeled'
            ),
        )
    reg = get_class_registry()
    try:
        result = reg.merge_class(payload.source_id, payload.target_id)
    except ClassRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target = reg.get(payload.target_id)
    target_name = target.class_name if target is not None else ''
    now = _now_iso()
    relabel_doc = {
        'class_id': payload.target_id,
        'class_name': target_name,
        'class_source': 'class_merge',
        'updated_at': now,
    }
    merge_query = {
        'bool': {
            'must': [{'term': {'class_id': payload.source_id}}],
            'must_not': [{'term': {'test_holdout': True}}],
        },
    }
    # The confirmed-labels index has no class_id_history / class_validated
    # field (its label_source means "original label provenance", not
    # "is this human-confirmed") — stays a plain update_by_query.
    try:
        await opensearch.update_by_query(
            index=CURATION_LABELS_CONFIRMED_INDEX,
            body={
                'script': {
                    'source': (
                        'ctx._source.class_id = params.class_id;'
                        'ctx._source.class_name = params.class_name;'
                        'ctx._source.class_source = params.class_source;'
                        'ctx._source.cluster_id = params.class_id;'
                        'ctx._source.remove("cluster_subid");'
                        'ctx._source.updated_at = params.updated_at;'
                    ),
                    'params': relabel_doc,
                },
                'query': merge_query,
            },
            conflicts='proceed',
            refresh=True,
        )
    except Exception as exc:
        logger.warning(
            'merge_relabel_failed', index=CURATION_LABELS_CONFIRMED_INDEX, error=str(exc)
        )

    # Items index: a per-doc OCC bulk pass (rather than a bare
    # update_by_query painless script) so class_id_history gets appended
    # (dedupe/cap both live in record_class_history, reused here rather
    # than reimplemented in painless). A merged crop must not keep a stale
    # class_validated=true/label_source='human' once class_source now says
    # 'class_merge'.
    from src.clients.occ import occ_skip_on_conflict_bulk
    from src.services.curation.history import record_class_history

    async def _scroll_merge_ids() -> list[str]:
        ids: list[str] = []
        body = {'size': 500, 'query': merge_query, '_source': False}
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body, scroll='2m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
        while hits:
            ids.extend(h['_id'] for h in hits)
            resp = await opensearch.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = resp.get('_scroll_id')
            hits = resp['hits']['hits']
        if scroll_id:
            try:
                await opensearch.clear_scroll(scroll_id=scroll_id)
            except Exception as exc:  # nosec B110 — advisory cleanup only
                logger.debug('merge_clear_scroll_failed', error=str(exc))
        return ids

    def _merge_crop(_doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        update = dict(relabel_doc)
        update['cluster_id'] = payload.target_id
        update['cluster_subid'] = None
        update['label_source'] = 'class_merge'
        update['class_validated'] = False
        update['class_id_history'] = record_class_history(current, writer='class_merge')
        return update

    try:
        crop_ids = await _scroll_merge_ids()
        if crop_ids:
            bulk_result = await occ_skip_on_conflict_bulk(
                opensearch,
                doc_ids=crop_ids,
                merger=_merge_crop,
                index=CURATION_ITEMS_INDEX,
                refresh=True,
                writer_id='class_merge',
            )
            if bulk_result.get('errors'):
                logger.warning(
                    'merge_relabel_partial_errors',
                    index=CURATION_ITEMS_INDEX,
                    errors=len(bulk_result['errors']),
                )
    except Exception as exc:
        logger.warning('merge_relabel_failed', index=CURATION_ITEMS_INDEX, error=str(exc))
    return result


@router.get('/classes/{class_id}/crops', response_model=CropsPageResponse)
async def class_crops(
    class_id: int,
    opensearch: OpenSearchDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    include_test: bool = False,
) -> CropsPageResponse:
    """Crops in a single class."""
    return await list_crops(
        opensearch=opensearch,
        page=page,
        page_size=page_size,
        class_id=class_id,
        include_test=include_test,
    )


@router.post('/classes/sync_to_opensearch')
async def sync_classes(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Rebuild the classes index from the on-disk registry."""
    return await get_class_registry().sync_to_opensearch(opensearch)
