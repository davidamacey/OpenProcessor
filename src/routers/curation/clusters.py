"""Curation router sub-module — cluster cards, representatives, refine, auto-promote."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException, Query

from src.routers.curation._api_error import api_error
from src.routers.curation._common import OpenSearchDep, items_index, router
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.schemas.curation.clusters import ClusterFacets, ClusterListResponse
from src.schemas.curation.config_common import ApiErrorResponse
from src.services.curation.cluster_cards import (
    ClusterListParams,
    ClusterSortId,
    ValidationFilter,
    fetch_representatives,
    list_cluster_cards,
)
from src.services.curation.cluster_facets import cluster_facets
from src.services.curation.cluster_purity import (
    PROMOTE_MIN_MEMBERS,
    PROMOTE_MIN_PRECISION,
    PROMOTE_MIN_PURITY,
)
from src.services.curation.clustering.refine import MAX_REFINE_MEMBERS
from src.services.curation.item_filter import ItemFilter, item_filter_clauses


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
    kind: Literal['class', 'candidate', 'all'] = Query('all'),
    cluster_id: int | None = Query(
        None,
        description=(
            'Filter to exactly this cluster_id. Distinct from a class: '
            'cluster_id == class_id is only an invariant for "class" '
            'clusters (id < RESIDUAL_CLUSTER_ID_OFFSET) — a "candidate" '
            'cluster (id >= that offset) has no matching class_id at all, '
            'so class_id can never be used to look one up by its own '
            'identity. This is the correct filter for "fetch one cluster '
            'card by id" regardless of kind.'
        ),
    ),
    # Primary-subject grid filters: every card stat (size, reps, purity)
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

    One OpenSearch aggregation yields every card input (size, validation,
    dominant class, label agreement, geometry purity, detector agreement and
    confidence, per-tab review counts, sub-clusters); the server then applies
    the filters and the ``sort`` and sets the quality ``flags``. A candidate
    cluster only gets a ``dominant_class_name`` when a unique top class has at
    least ``CANDIDATE_DOMINANT_MIN_COUNT`` members and
    ``CANDIDATE_DOMINANT_MIN_SHARE`` of the labelled ones. ``truncated`` is
    true when the aggregation returned ``max_clusters`` buckets: the list may
    then be incomplete. Frozen test-holdout and excluded items are never counted.
    """
    try:
        gate = ItemFilter(
            **item_filter.model_dump(exclude_defaults=True),
            min_blur_ratio=min_blur_ratio,
            class_source=class_source,
        )
        item_filter_clauses(gate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await list_cluster_cards(
        opensearch,
        ClusterListParams(
            item_filter=gate,
            per_cluster=per_cluster,
            max_clusters=max_clusters,
            kind=kind,
            cluster_id=cluster_id,
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
    '/clusters/{cluster_id}/facets',
    response_model=None,
    responses={200: {'model': ClusterFacets}},
)
async def cluster_facets_endpoint(cluster_id: int, opensearch: OpenSearchDep) -> ClusterFacets:
    """Counts of one cluster's members by detector class, class source, validation,
    review tab and detector confidence, for filter chips. Each count is the ``total`` of
    the filtered ``GET /crops`` call named in :class:`ClusterFacets`. A cluster with no
    members answers zeros."""
    return await cluster_facets(opensearch, cluster_id)


@router.get('/clusters/representatives')
async def cluster_representatives(
    opensearch: OpenSearchDep,
    per_cluster: int = Query(4, ge=1, le=10, description='Top crops per cluster'),
    max_clusters: int = Query(200, ge=1, le=1000),
    class_id: int | None = Query(None, description='Restrict to clusters containing this class'),
    offset: int = Query(0, ge=0, description='Offset into the _count-desc cluster list'),
) -> dict[str, Any]:
    """Return up to ``per_cluster`` representative crop_ids per cluster, one page.

    The cluster id list still comes from one cheap ``terms``
    aggregation (no sub-agg), but representatives are fetched via one
    ``_msearch`` covering only the ``[offset, offset+max_clusters)`` page
    of cluster ids (ordered by member count desc) — not via a
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
                    'field': 'cluster_id',
                    'size': offset + max_clusters,
                    'order': {'_count': 'desc'},
                },
            },
        },
    }
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'representatives query failed: {exc}') from exc

    buckets = resp.get('aggregations', {}).get('clusters', {}).get('buckets', [])
    page_cluster_ids: list[int] = [int(b['key']) for b in buckets[offset : offset + max_clusters]]

    reps = await fetch_representatives(opensearch, page_cluster_ids, per_cluster)
    out: dict[str, Any] = {
        str(cid): card_reps for cid, card_reps in zip(page_cluster_ids, reps, strict=True)
    }

    return {
        'clusters': out,
        'count': len(out),
        'offset': offset,
        'max_clusters': max_clusters,
    }


@router.post('/clusters/refine/{cluster_id}')
async def refine_cluster_endpoint(
    cluster_id: int,
    opensearch: OpenSearchDep,
    distance_threshold: float = Query(0.25, ge=0.05, le=1.0),
    max_members: int = Query(
        MAX_REFINE_MEMBERS,
        ge=50,
        le=50000,
        description=(
            'Skip clusters larger than this. AHC builds a full ~8*n^2-byte '
            'pairwise matrix, so larger values cost RAM (8000~512MB, 12000~1.15GB). '
            f'Default {MAX_REFINE_MEMBERS} (env OP_MAX_REFINE_MEMBERS).'
        ),
    ),
) -> dict[str, Any]:
    """Run AHC refinement on a curation item cluster.

    Writes ``cluster_subid`` (keyword, e.g. ``"47a"``) to every member.
    Skips clusters below the small-cluster floor or above ``max_members``
    (the pairwise matrix grows ~8*n^2 bytes). Re-running on the same
    cluster overwrites prior subids, so refinement is idempotent.
    """
    from src.services.curation.clustering.refine import refine_cluster

    return await refine_cluster(
        opensearch,
        cluster_id,
        distance_threshold=distance_threshold,
        max_members=max_members,
    )


@router.post('/clusters/auto_promote', responses={409: {'model': ApiErrorResponse}})
async def auto_promote_clusters_endpoint(
    opensearch: OpenSearchDep,
    min_purity: float = Query(
        PROMOTE_MIN_PURITY, ge=0.5, le=1.0, description='Min dominant-class share'
    ),
    min_members: int = Query(
        PROMOTE_MIN_MEMBERS, ge=2, le=1000, description='Skip clusters smaller than this'
    ),
    dry_run: bool = Query(False, description='Compute summary without writing'),
    promote_min_precision: float = Query(
        PROMOTE_MIN_PRECISION,
        ge=0.0,
        le=1.0,
        description='Audited detector precision a class needs to be promoted',
    ),
    force: bool = Query(False, description='Promote without the accuracy-audit gate (logged)'),
) -> dict[str, Any]:
    """Promote classifier-labelled crops that agree with their high-purity cluster.

    Validates each such crop (``class_source='cluster_majority_agreement'``).
    Gated on the accuracy audit (``POST /audit/start``): every class the run would
    promote needs enough audited crops and an audited detector precision of at
    least ``promote_min_precision``, else ``409 audit_required`` or
    ``409 audit_precision_low`` listing the classes and nothing is written.
    ``force=true`` bypasses the gate; a dry run is never gated.
    """
    from src.services.curation.clustering.auto_promote import AuditGateError, gated_auto_promote

    try:
        return await gated_auto_promote(
            opensearch,
            min_purity=min_purity,
            min_members=min_members,
            dry_run=dry_run,
            promote_min_precision=promote_min_precision,
            force=force,
        )
    except AuditGateError as exc:
        raise api_error(
            409,
            exc.code,
            f'accuracy audit does not clear {len(exc.classes)} class(es) for promotion; '
            'audit them or pass force=true',
            classes=exc.classes,
            promote_min_precision=exc.threshold,
        ) from exc
