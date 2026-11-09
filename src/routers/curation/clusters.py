"""Curation router sub-module — cluster cards, facets, representatives."""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves the path and query types at runtime
from typing import Any

from fastapi import HTTPException, Query

from src.routers._errors import server_error
from src.routers.curation._common import OpenSearchDep, items_index, router
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.schemas.curation.clusters import ClusterFacets, ClusterListResponse
from src.services.curation.cluster_cards import (
    ClusterListParams,
    ClusterSortId,
    ValidationFilter,
    fetch_representatives,
    list_cluster_cards,
)
from src.services.curation.cluster_facets import cluster_facets
from src.services.curation.item_filter import ItemFilter, item_filter_clauses
from src.services.embedding_spaces.readers import item_write_fields


@router.get('/clusters', response_model=None, responses={200: {'model': ClusterListResponse}})
async def list_clusters(
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    per_cluster: int = Query(4, ge=0, le=10, description='Representative crops per cluster'),
    max_clusters: int = Query(
        1000,
        ge=1,
        le=10000,
        description='Clusters the aggregation returns; `truncated` says when it was cut here.',
    ),
    cluster_uid: uuid.UUID | None = Query(
        None,
        description='Return only the card of this cluster (its identity in the served partition).',
    ),
    # Primary-subject grid filters: every card stat (size, reps)
    # reflects only crops passing these.
    min_blur_ratio: float | None = Query(None, ge=0.0),
    class_source: str | None = Query(None),
    sort: ClusterSortId = Query('size_desc', description='Order of the served cards.'),
    dominant_class: list[str] | None = Query(None, description='Dominant class name, repeatable.'),
    detector_class: list[str] | None = Query(
        None, description="The detector's top class in the cluster, repeatable."
    ),
    min_size: int | None = Query(None, ge=0),
    max_size: int | None = Query(None, ge=0),
    validation_state: ValidationFilter | None = Query(None),
    flag: list[str] | None = Query(None, description='Cards carrying any of these flags.'),
    needs_review: bool | None = Query(
        None, description='True: cards with members in the review queue; false: those without.'
    ),
    # Representatives are fetched only for this page of the served (filtered,
    # sorted) card list; every card up to max_clusters is still returned, with
    # an empty ``representatives`` list outside the page.
    offset: int = Query(0, ge=0, description='Offset into the served cards for representatives'),
    limit: int = Query(50, ge=1, le=500, description='Cards to compute representatives for'),
) -> ClusterListResponse:
    """Authoritative per-cluster card list for the labeler.

    One OpenSearch aggregation over the served partition's ``cluster_uid`` yields every card
    input (size, validation, dominant class, label agreement, detector agreement and
    confidence, per-tab review counts); the server then applies the filters and the
    ``sort`` and sets the quality ``flags``. ``truncated`` is
    true when the aggregation returned ``max_clusters`` buckets: the list may
    then be incomplete. Frozen test-holdout and excluded items are never counted.
    """
    try:
        gate = ItemFilter(
            **item_filter.model_dump(exclude_defaults=True),
            min_blur_ratio=min_blur_ratio,
            class_source=class_source,
        )
        item_filter_clauses(gate, embedding=await item_write_fields())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await list_cluster_cards(
        opensearch,
        ClusterListParams(
            item_filter=gate,
            per_cluster=per_cluster,
            max_clusters=max_clusters,
            cluster_uid=str(cluster_uid) if cluster_uid is not None else None,
            offset=offset,
            limit=limit,
            sort=sort,
            dominant_class=tuple(dominant_class or ()),
            detector_class=tuple(detector_class or ()),
            min_size=min_size,
            max_size=max_size,
            validation_state=validation_state,
            flags=tuple(flag or ()),
            needs_review=needs_review,
        ),
    )


@router.get(
    '/clusters/{cluster_uid}/facets',
    response_model=None,
    responses={200: {'model': ClusterFacets}},
)
async def cluster_facets_endpoint(
    cluster_uid: uuid.UUID, opensearch: OpenSearchDep
) -> ClusterFacets:
    """Counts of one cluster's members by detector class, class source, validation,
    review tab and detector confidence, for filter chips. Each count is the ``total`` of
    the filtered ``GET /crops`` call named in :class:`ClusterFacets`. A cluster with no
    members answers zeros."""
    return await cluster_facets(opensearch, str(cluster_uid))


_REPRESENTATIVES_MAX_CLUSTERS = 1000
# The terms aggregation asks for offset + max_clusters buckets; OpenSearch refuses more than
# search.max_buckets (default 65,535): https://docs.opensearch.org/latest/aggregations/bucket/index/
_OPENSEARCH_MAX_BUCKETS = 65_535
_REPRESENTATIVES_MAX_OFFSET = _OPENSEARCH_MAX_BUCKETS - _REPRESENTATIVES_MAX_CLUSTERS


@router.get('/clusters/representatives')
async def cluster_representatives(
    opensearch: OpenSearchDep,
    per_cluster: int = Query(4, ge=1, le=10, description='Top crops per cluster'),
    max_clusters: int = Query(200, ge=1, le=_REPRESENTATIVES_MAX_CLUSTERS),
    class_id: int | None = Query(None, description='Restrict to clusters containing this class'),
    offset: int = Query(
        0,
        ge=0,
        le=_REPRESENTATIVES_MAX_OFFSET,
        description='Offset into the _count-desc cluster list',
    ),
) -> dict[str, Any]:
    """Return up to ``per_cluster`` representative crop_ids per cluster, one page.

    The cluster uid list still comes from one cheap ``terms``
    aggregation (no sub-agg), but representatives are fetched via one
    ``_msearch`` covering only the ``[offset, offset+max_clusters)`` page
    of cluster uids (ordered by member count desc) — not via a
    ``top_hits`` sub-agg that would decompress stored ``_source`` for
    every representative across every bucket. Only clusters in that page
    appear as keys in the response; call again with a larger ``offset``
    for the next page.

    When ``class_id`` is set, the outer query filters to crops with that
    class so the agg only returns clusters that contain at least one
    member of the class — driving the labeler's class-sidebar filter.
    """
    # Excluded items must not surface as cluster representatives
    # either -- this endpoint had no class_excluded guard at all before.
    query: dict[str, Any] = {
        'bool': {
            'filter': [{'term': {'class_id': class_id}}] if class_id is not None else [],
            'must_not': [{'term': {'class_excluded': True}}],
        }
    }
    body = {
        'size': 0,
        'query': query,
        'aggs': {
            'clusters': {
                'terms': {
                    'field': 'cluster_uid',
                    'size': offset + max_clusters,
                    'order': {'_count': 'desc'},
                },
            },
        },
    }
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except Exception as exc:
        raise server_error(500, 'representatives query failed', exc) from exc

    buckets = resp.get('aggregations', {}).get('clusters', {}).get('buckets', [])
    page_cluster_uids: list[str] = [str(b['key']) for b in buckets[offset : offset + max_clusters]]

    reps = await fetch_representatives(opensearch, page_cluster_uids, per_cluster)
    out: dict[str, Any] = dict(zip(page_cluster_uids, reps, strict=True))

    return {
        'clusters': out,
        'count': len(out),
        'offset': offset,
        'max_clusters': max_clusters,
    }
