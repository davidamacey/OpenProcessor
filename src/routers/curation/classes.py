"""Curation router sub-module — class registry reads: sources, list, one class."""

from __future__ import annotations

from fastapi import Depends, HTTPException

from src.routers._errors import server_error
from src.routers.curation._class_hotkeys import project_reserved_hotkeys
from src.routers.curation._class_models import ClassEntry, ClassListResponse
from src.routers.curation._common import (
    OpenSearchDep,
    _raw_opensearch_dep,
    get_class_registry,
    router,
)
from src.services.curation.class_counts import fetch_class_counts
from src.services.curation.class_sources import class_source_catalog
from src.services.curation.dataset_thresholds import adequacy, dataset_thresholds


@router.get('/class_sources', dependencies=[Depends(_raw_opensearch_dep)])
async def list_class_sources() -> dict[str, list[dict[str, str]]]:
    """Every ``class_source`` value this deployment can write.

    ``{"class_sources": [{"id", "label", "role"}, ...]}``. Ingest values
    follow ``OP_INGEST_PRIMARY_*`` / ``OP_INGEST_SECONDARY_*``; the rest
    are fixed writer values. ``role`` is one of
    :data:`~src.services.curation.class_sources.CLASS_SOURCE_ROLES`.
    """
    return {'class_sources': class_source_catalog()}


@router.get('/classes', response_model=ClassListResponse)
async def list_classes(opensearch: OpenSearchDep) -> ClassListResponse:
    """List every class in the registry, with live counts from the items index."""
    reg = await get_class_registry().snapshot()
    try:
        counts, _aggs = await fetch_class_counts(opensearch)
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc

    # A region-of-interest class (e.g. the active region profile's
    # region_class_name) lives as a sub-bbox on every parent item that has
    # one, NOT as a separate doc whose primary class_id == that class. The
    # class-aggregation count above only captures the rare mis-labels.
    #
    # This used to override sample_count/validated_count
    # with the region inventory total, which made the region "class" look
    # like an item class with thousands of validated crops -- it inflated
    # /train's class picker and /export's per-class table (the served
    # sample_count disagreed with the real item-crop count everywhere
    # else). The region inventory is already served separately (GET
    # /regions, /regions/statuses, stats/dataset). Here we only mark
    # ``kind='region'`` so item-count consumers can exclude it; item
    # counts stay the real (usually zero) class-aggregation numbers.
    from src.services.curation.region_class import is_region_class

    thresholds = dataset_thresholds()
    hard_min = int(thresholds.get('block_below', 0))

    return ClassListResponse(
        classes=[
            ClassEntry(
                class_id=c.class_id,
                class_name=c.class_name,
                group=c.group,
                sample_count=counts.count.get(c.class_id, 0),
                validated_count=counts.validated.get(c.class_id, 0),
                deprecated=c.deprecated,
                merged_into=getattr(c, 'merged_into', None),
                hotkey_letter=getattr(c, 'hotkey_letter', None),
                adequacy=adequacy(counts.validated.get(c.class_id, 0)),
                added_at=getattr(c, 'added_at', None),
                kind='region' if is_region_class(c) else 'item',
                trainable=counts.trainable(c.class_id),
                trainable_gap=max(0, hard_min - counts.trainable(c.class_id)),
            )
            for c in reg.classes
        ],
        thresholds=thresholds,
        reserved_hotkeys=await project_reserved_hotkeys(),
    )


@router.get('/classes/{class_id}', response_model=ClassEntry)
async def get_class(class_id: int, opensearch: OpenSearchDep) -> ClassEntry:
    """One registry class with the same live counts ``GET /classes`` reports."""
    for entry in (await list_classes(opensearch)).classes:
        if entry.class_id == class_id:
            return entry
    raise HTTPException(status_code=404, detail=f'unknown class_id {class_id}')
