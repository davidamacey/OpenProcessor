"""Curation router sub-module — class registry reads: sources, list, one class."""

from __future__ import annotations

from fastapi import HTTPException

from src.routers.curation._class_hotkeys import project_reserved_hotkeys
from src.routers.curation._class_models import ClassEntry, ClassListResponse
from src.routers.curation._common import OpenSearchDep, get_class_registry, items_index, router
from src.services.curation.class_sources import class_source_catalog
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.dataset_thresholds import adequacy, dataset_thresholds


@router.get('/class_sources')
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
    validated_test_holdout: dict[int, int] = {}
    validated_excluded: dict[int, int] = {}
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
                        # Trainable = validated minus these two (frozen
                        # test-holdout, human-excluded) -- see ClassEntry.trainable.
                        'validated_test_holdout': {
                            'filter': {
                                'bool': {
                                    'filter': [
                                        {'term': {'class_validated': True}},
                                        {'term': {'test_holdout': True}},
                                    ]
                                }
                            }
                        },
                        'validated_excluded': {
                            'filter': {
                                'bool': {
                                    'filter': [
                                        {'term': {'class_validated': True}},
                                        {'term': {'class_excluded': True}},
                                    ]
                                }
                            }
                        },
                    },
                },
                # Restrict to class-kind cluster ids (cluster_id ==
                # class_id by invariant) via a filter agg before
                # terms-aggregating, so candidate/residual cluster ids
                # (>= RESIDUAL_CLUSTER_ID_OFFSET, often far more numerous
                # than the ~80 class ids) can't crowd real class buckets
                # out of the size-1000 cap.
                'by_cluster': {
                    'filter': {
                        'range': {'cluster_id': {'gte': 0, 'lt': RESIDUAL_CLUSTER_ID_OFFSET}}
                    },
                    'aggs': {
                        'classes': {
                            'terms': {'field': 'cluster_id', 'size': 1000},
                        },
                    },
                },
            },
        }
        resp = await opensearch.search(index=items_index(), body=body)
        aggs = resp.get('aggregations') or {}
        for bucket in aggs.get('by_class', {}).get('buckets', []):
            cid = int(bucket['key'])
            counts[cid] = int(bucket.get('doc_count', 0))
            validated[cid] = int((bucket.get('validated') or {}).get('doc_count', 0))
            validated_test_holdout[cid] = int(
                (bucket.get('validated_test_holdout') or {}).get('doc_count', 0)
            )
            validated_excluded[cid] = int(
                (bucket.get('validated_excluded') or {}).get('doc_count', 0)
            )
        by_cluster_buckets = (aggs.get('by_cluster') or {}).get('classes', {}).get('buckets', [])
        for bucket in by_cluster_buckets:
            cid = int(bucket['key'])
            cluster_size[cid] = int(bucket.get('doc_count', 0))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    # A region-of-interest class (e.g. the active region profile's
    # region_class_name) lives as a sub-bbox on every parent item that has
    # one, NOT as a separate doc whose primary class_id == that class. The
    # class-aggregation count above only captures the rare mis-labels.
    #
    # This used to override sample_count/validated_count/cluster_size
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

    def _trainable(cid: int, n_valid: int) -> int:
        return n_valid - validated_test_holdout.get(cid, 0) - validated_excluded.get(cid, 0)

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
                merged_into=getattr(c, 'merged_into', None),
                hotkey_letter=getattr(c, 'hotkey_letter', None),
                adequacy=adequacy(validated.get(c.class_id, c.validated_count)),
                added_at=getattr(c, 'added_at', None),
                kind='region' if is_region_class(c) else 'item',
                trainable=_trainable(c.class_id, validated.get(c.class_id, c.validated_count)),
                trainable_gap=max(
                    0,
                    hard_min - _trainable(c.class_id, validated.get(c.class_id, c.validated_count)),
                ),
            )
            for c in reg.classes
        ],
        thresholds=thresholds,
        reserved_hotkeys=await project_reserved_hotkeys(opensearch),
    )


@router.get('/classes/{class_id}', response_model=ClassEntry)
async def get_class(class_id: int, opensearch: OpenSearchDep) -> ClassEntry:
    """One registry class with the same live counts ``GET /classes`` reports."""
    for entry in (await list_classes(opensearch)).classes:
        if entry.class_id == class_id:
            return entry
    raise HTTPException(status_code=404, detail=f'unknown class_id {class_id}')
