"""Curation router sub-module — cluster cards, representatives, refine, auto-promote."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException, Query

from src.routers.curation._common import CURATION_ITEMS_INDEX, OpenSearchDep, router
from src.services.curation.cluster_ids import (
    CORE_SIMILARITY_MIN,
    RESIDUAL_CLUSTER_ID_OFFSET,
    cluster_kind,
)
from src.services.curation.cluster_purity import (
    PROMOTE_MIN_MEMBERS,
    PROMOTE_MIN_PURITY,
    is_promotable,
    purity_thresholds,
    purity_tier,
)
from src.services.curation.clustering.orchestrator import MAX_REFINE_MEMBERS


CANDIDATE_DOMINANT_MIN_COUNT = 3
"""A candidate cluster names a dominant class only when at least this many
members carry it."""

CANDIDATE_DOMINANT_MIN_SHARE = 0.5
"""...and that class holds at least this share of the labelled members."""


def _candidate_dominant_name(cls_buckets: list[dict[str, Any]], labelled: int) -> str | None:
    """Dominant class name for a candidate cluster card, or ``None``.

    A candidate cluster is an unnamed residual grouping; naming it after
    the plurality class of a handful of labelled members (e.g. a 1-1-1
    tie in a 116-member cluster) overclaims. Require a unique winner that
    clears both floors.
    """
    if not cls_buckets or labelled <= 0:
        return None
    top = int(cls_buckets[0]['doc_count'])
    runner_up = int(cls_buckets[1]['doc_count']) if len(cls_buckets) > 1 else 0
    if (
        top >= CANDIDATE_DOMINANT_MIN_COUNT
        and top / labelled >= CANDIDATE_DOMINANT_MIN_SHARE
        and top > runner_up
    ):
        return str(cls_buckets[0]['key'])
    return None


@router.get('/clusters')
async def list_clusters(
    opensearch: OpenSearchDep,
    per_cluster: int = Query(4, ge=0, le=10, description='Representative crops per cluster'),
    max_clusters: int = Query(1000, ge=1, le=10000),
    kind: Literal['class', 'candidate', 'all'] = Query('all'),
    class_id: int | None = Query(None, description='Filter to clusters containing this class'),
    cluster_id: int | None = Query(
        None,
        description=(
            'Filter to exactly this cluster_id. Distinct from class_id: '
            'cluster_id == class_id is only an invariant for "class" '
            'clusters (id < RESIDUAL_CLUSTER_ID_OFFSET) — a "candidate" '
            'cluster (id >= that offset) has no matching class_id at all, '
            'so class_id can never be used to look one up by its own '
            'identity. This is the correct filter for "fetch one cluster '
            'card by id" regardless of kind.'
        ),
    ),
    # Primary-subject grid filters: every card stat (size, reps, purity)
    # reflects only crops passing these — so a filtered grid shows clusters
    # of just the largest/clear crops, ready to drag-drop + AHC-refine.
    max_rank: int | None = Query(None, ge=1),
    min_blur_ratio: float | None = Query(None, ge=0.0),
    class_source: str | None = Query(None),
) -> dict[str, Any]:
    """Authoritative per-cluster card payload for the labeler.

    Single OpenSearch aggregation that returns, for every cluster_id:

    * ``size`` (total members), ``validated_count`` (class_validated=true),
    * ``dominant_class_{id,name,count}`` and ``purity`` (largest-class
      share among labelled members). A candidate cluster only gets a
      ``dominant_class_name`` when a unique top class has at least
      ``CANDIDATE_DOMINANT_MIN_COUNT`` members and
      ``CANDIDATE_DOMINANT_MIN_SHARE`` of the labelled ones;
      ``dominant_count``/``labelled_count``/``purity`` are always reported,
    * ``is_unlabeled`` (true when no class_name has any signal at all),
    * ``cluster_kind`` (``class`` | ``candidate`` | ``unassigned``),
    * ``n_subclusters`` (distinct cluster_subid values), and
    * ``representatives`` (top ``per_cluster`` crops sorted by
      cluster_distance asc).

    This endpoint exists so the frontend can stop computing dominant
    class / purity from a 4-crop sample (it was getting wrong answers
    on candidate clusters where all reps were null-class). The
    aggregation is cheap because ``cluster_id`` is a low-cardinality
    integer field; ``class_name``/``class_source``/``cluster_subid`` are
    all mapped ``keyword`` directly on the live index (no ``.keyword``
    subfield).
    """
    # Human-ignored crops (class_excluded=true) must not contribute to
    # any cluster card — they've left their candidate bucket and should
    # never resurface for the operator to re-sort.
    _base_match: dict[str, Any]
    if cluster_id is not None:
        _base_match = {'term': {'cluster_id': cluster_id}}
    elif class_id is not None:
        _base_match = {'term': {'class_id': class_id}}
    else:
        _base_match = {'match_all': {}}
    # Primary-subject gate (filter context → cached, scopes every sub-agg).
    # Null-safe blur: crops with no blur score aren't hidden by the slider.
    gate_filter: list[dict[str, Any]] = []
    if max_rank is not None:
        gate_filter.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})
    if min_blur_ratio is not None:
        gate_filter.append(
            {
                'bool': {
                    'should': [
                        {'range': {'blur_lap_ratio': {'gte': min_blur_ratio}}},
                        {'bool': {'must_not': {'exists': {'field': 'blur_lap_ratio'}}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
    if class_source:
        # class_source is mapped keyword directly on the live index — no
        # .keyword subfield exists (same root cause as top_class below).
        gate_filter.append({'term': {'class_source': class_source}})
    outer_query: dict[str, Any] = {
        'bool': {
            'must': [_base_match],
            'must_not': [{'term': {'class_excluded': True}}],
            **({'filter': gate_filter} if gate_filter else {}),
        }
    }
    cluster_terms_agg: dict[str, Any] = {
        'terms': {
            'field': 'cluster_id',
            'size': max_clusters,
            'order': {'_count': 'desc'},
        },
        'aggs': {
            'top_class': {
                'terms': {
                    # `class_name` is already mapped `keyword` on the live
                    # index (not `text`) — there is no `.keyword` subfield
                    # to query.
                    'field': 'class_name',
                    'size': 3,
                    'order': {'_count': 'desc'},
                },
            },
            # True labelled count: top_class only returns its top buckets,
            # so summing them undercounts clusters with many classes.
            'labelled': {'filter': {'exists': {'field': 'class_name'}}},
            'validated': {'filter': {'term': {'class_validated': True}}},
            # cluster_subid is mapped keyword directly on the live index —
            # no .keyword subfield exists.
            'subclusters': {'cardinality': {'field': 'cluster_subid'}},
            'latest_update': {'max': {'field': 'updated_at'}},
        },
    }
    if per_cluster > 0:
        cluster_terms_agg['aggs']['reps'] = {
            'top_hits': {
                'size': per_cluster,
                'sort': [
                    {
                        'cluster_distance': {
                            'order': 'asc',
                            'missing': '_last',
                            'unmapped_type': 'double',
                        },
                    },
                ],
                '_source': [
                    'crop_id',
                    'cluster_distance',
                    'class_name',
                    'cluster_subid',
                ],
            },
        }
    body = {'size': 0, 'query': outer_query, 'aggs': {'clusters': cluster_terms_agg}}
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'clusters query failed: {exc}') from exc

    items: list[dict[str, Any]] = []
    total_class = 0
    total_candidate = 0
    for bucket in resp.get('aggregations', {}).get('clusters', {}).get('buckets', []):
        cid = int(bucket['key'])
        ck = cluster_kind(cid) or 'unassigned'
        if kind not in ('all', ck):
            continue
        size = int(bucket['doc_count'])
        cls_buckets = bucket.get('top_class', {}).get('buckets', [])
        labelled_total = sum(int(b['doc_count']) for b in cls_buckets)
        if 'labelled' in bucket:
            labelled_total = max(labelled_total, int(bucket['labelled'].get('doc_count') or 0))
        top_name: str | None = None
        top_count = 0
        if cls_buckets:
            top_name = cls_buckets[0]['key']
            top_count = int(cls_buckets[0]['doc_count'])
        purity = (top_count / labelled_total) if labelled_total else None
        if ck == 'candidate':
            top_name = _candidate_dominant_name(cls_buckets, labelled_total)
        is_unlabeled = labelled_total == 0
        validated_count = int(bucket.get('validated', {}).get('doc_count') or 0)
        n_subclusters = int(bucket.get('subclusters', {}).get('value') or 0)
        # The dominant class id is the v6 registry id for top_name when the
        # cluster is a class cluster (cluster_id == class_id by invariant);
        # candidate clusters have no class id yet so it stays None.
        dominant_class_id: int | None = cid if ck == 'class' and top_name else None
        reps: list[dict[str, Any]] = []
        for h in bucket.get('reps', {}).get('hits', {}).get('hits', []) or []:
            src = h.get('_source') or {}
            crop_id = src.get('crop_id') or h.get('_id')
            if crop_id:
                reps.append(
                    {
                        'crop_id': crop_id,
                        'cluster_distance': src.get('cluster_distance'),
                        'class_name': src.get('class_name'),
                        'cluster_subid': src.get('cluster_subid'),
                    },
                )
        items.append(
            {
                'cluster_id': cid,
                'cluster_kind': ck,
                'size': size,
                'validated_count': validated_count,
                'labelled_count': labelled_total,
                'dominant_class_id': dominant_class_id,
                'dominant_class_name': top_name,
                'dominant_count': top_count,
                'purity': purity,
                # Same thresholds as the auto-promote gate.
                'purity_tier': purity_tier(purity),
                'promotable': is_promotable(members=size, labelled=labelled_total, purity=purity),
                'is_unlabeled': is_unlabeled,
                'n_subclusters': n_subclusters,
                'updated_at': bucket.get('latest_update', {}).get('value_as_string'),
                'representatives': reps,
            },
        )
        if ck == 'candidate':
            total_candidate += 1
        elif ck == 'class':
            total_class += 1
    return {
        'items': items,
        'total': len(items),
        'total_class_clusters': total_class,
        'total_candidate_clusters': total_candidate,
        'cluster_id_offset': RESIDUAL_CLUSTER_ID_OFFSET,
        'purity_thresholds': purity_thresholds(),
        'core_similarity_min': CORE_SIMILARITY_MIN,
    }


@router.get('/clusters/representatives')
async def cluster_representatives(
    opensearch: OpenSearchDep,
    per_cluster: int = Query(4, ge=1, le=10, description='Top crops per cluster'),
    max_clusters: int = Query(200, ge=1, le=1000),
    class_id: int | None = Query(None, description='Restrict to clusters containing this class'),
) -> dict[str, Any]:
    """Return up to ``per_cluster`` representative crop_ids for every cluster.

    Single OpenSearch query using ``terms`` aggregation on ``cluster_id`` with a
    ``top_hits`` sub-aggregation sorted by ascending cluster_distance (i.e.,
    closest to centroid first). Designed for the labeler's clusters grid so it
    can render all card thumbnails without N round-trips.

    When ``class_id`` is set, the outer query filters to crops with that
    class so the agg only returns clusters that contain at least one
    member of the class — driving the labeler's class-sidebar filter.
    """
    query: dict[str, Any] = (
        {'term': {'class_id': class_id}} if class_id is not None else {'match_all': {}}
    )
    body = {
        'size': 0,
        'query': query,
        'aggs': {
            'clusters': {
                'terms': {
                    'field': 'cluster_id',
                    'size': max_clusters,
                    'order': {'_count': 'desc'},
                },
                'aggs': {
                    'reps': {
                        'top_hits': {
                            'size': per_cluster,
                            'sort': [
                                {
                                    'cluster_distance': {
                                        'order': 'asc',
                                        'missing': '_last',
                                        'unmapped_type': 'double',
                                    }
                                }
                            ],
                            '_source': [
                                'crop_id',
                                'cluster_distance',
                                'class_name',
                                'cluster_subid',
                            ],
                        },
                    },
                },
            },
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'representatives query failed: {exc}') from exc

    out: dict[str, Any] = {}
    for bucket in resp.get('aggregations', {}).get('clusters', {}).get('buckets', []):
        cid = int(bucket['key'])
        hits = bucket.get('reps', {}).get('hits', {}).get('hits', [])
        crops = []
        for h in hits:
            src = h.get('_source') or {}
            crop_id = src.get('crop_id') or h.get('_id')
            if crop_id:
                crops.append(
                    {
                        'crop_id': crop_id,
                        'cluster_distance': src.get('cluster_distance'),
                        'class_name': src.get('class_name'),
                        'cluster_subid': src.get('cluster_subid'),
                    }
                )
        out[str(cid)] = crops
    return {'clusters': out, 'count': len(out)}


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
    """Run AHC refinement on a curation "vehicles" cluster.

    Writes ``cluster_subid`` (keyword, e.g. ``"47a"``) to every member.
    Skips clusters below the small-cluster floor or above ``max_members``
    (the pairwise matrix grows ~8*n^2 bytes). Re-running on the same
    cluster overwrites prior subids, so refinement is idempotent.
    """
    from src.services.curation.clustering.orchestrator import refine_cluster

    return await refine_cluster(
        opensearch,
        cluster_id,
        distance_threshold=distance_threshold,
        max_members=max_members,
    )


@router.post('/clusters/auto_promote')
async def auto_promote_clusters_endpoint(
    opensearch: OpenSearchDep,
    min_purity: float = Query(
        PROMOTE_MIN_PURITY, ge=0.5, le=1.0, description='Min dominant-class share'
    ),
    min_members: int = Query(
        PROMOTE_MIN_MEMBERS, ge=2, le=1000, description='Skip clusters smaller than this'
    ),
    dry_run: bool = Query(False, description='Compute summary without writing'),
) -> dict[str, Any]:
    """Promote crops in high-purity clusters to ``label_validated=true``.

    Sets ``label_source='cluster_propagation'`` and copies the dominant
    cluster class to every member that isn't already human-validated.
    """
    from src.services.curation.clustering.orchestrator import auto_promote_clusters

    return await auto_promote_clusters(
        opensearch,
        min_purity=min_purity,
        min_members=min_members,
        dry_run=dry_run,
    )
