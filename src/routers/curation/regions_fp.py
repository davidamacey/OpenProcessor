"""Curation region-clustering + false-positive endpoints.

Split out of ``regions.py`` to keep each router sub-module focused (and
under the 700-LOC gate). Covers the region-cluster *view* (coarse KMeans
buckets + AHC refine), the permanent false-positive bucket, and the
FP-centroid matcher.

False positives are heterogeneous (background clutter, similar-looking
non-target objects, empty/spurious detections), so the matcher is
**double-layered**:
:func:`~src.services.curation.clustering.region_box_clustering.build_region_fp_centroids`
sub-types the false-positive *boxes* into ``k`` sub-clusters and persists
**one centroid per sub-type**; :func:`suspected_false_positives` matches
each candidate box's vector against *all* sub-centroids and keeps the
nearest (one row per box). A new FP that doesn't resemble the bucket average is still
caught by whichever sub-type it's closest to. Re-run the build after
marking a batch of FPs to retrain the sub-centroids.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import HTTPException, Query

from src.config import get_region_fields
from src.config.project_context import current_project, project_api_base
from src.config.region_state import RegionStatus
from src.routers.curation._common import (
    OpenSearchDep,
    RegionProfileDep,
    _ensure_indexes,
    items_index,
    logger,
    router,
)
from src.routers.curation._error_models import REGION_PROFILE_RESPONSES
from src.routers.curation._region_row_models import RegionRowPage
from src.routers.curation.regions import _REGION_SOURCE_EXCLUDES
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.clustering.region_box_rows import box_state_clause
from src.services.curation.region_rows import rows_for_pairs
from src.services.curation.wire import box_thumbnail_url


# ``dominant_class_name`` for a non-FP region-cluster card. Region clusters
# are single-class by construction (every member is a region), so the card
# carries the generic region name rather than a domain class.
REGION_CLUSTER_CLASS_NAME = 'region'


SUSPECTED_FP_MAX_DISTANCE = 0.35
"""Default max distance to the nearest FP sub-centroid for a region to be
suggested as a false positive (served as ``default_threshold``)."""


_SUSPECTED_FP_CACHE_TTL_SEC = 60.0
"""Interim fix: ``/regions/suspected_false_positives`` scrolled the
entire region-embedding pool on every single page request (the pool is
independent of ``page``/``page_size``). Cache the scored
``[(dist, crop_id, box_id, subid)]`` list keyed by ``(project, trained_at, threshold)`` for
60s so paging through results doesn't re-scroll. The persisted-write
version (store ``region_fp_distance`` at write time) is the long-term
fix but is out of scope here."""

_suspected_fp_cache: dict[
    tuple[str, Any, float], tuple[float, list[tuple[float, str, str, str | None]]]
] = {}


@router.post('/regions/cluster', responses=REGION_PROFILE_RESPONSES)
async def cluster_regions(
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
    max_rank: int | None = Query(None, ge=1, description='Only top-N largest crops.'),
    auto_fp_threshold: float = Query(
        0.20,
        ge=0.0,
        le=1.0,
        description='Auto-move regions within this L2 distance of an FP sub-centroid '
        'into the FP bucket (tight, near-certain matches). 0 disables.',
    ),
    rebuild_fp_centroids: bool = Query(
        True, description='Rebuild FP sub-type centroids before the auto-assign pass.'
    ),
    force_repartition: bool = Query(
        False,
        description='Re-partition the good regions even if a manual refine happened '
        'recently. Default respects the refine TTL so recent sub-clusters are kept.',
    ),
) -> dict[str, Any]:
    """Launch the full region-clustering pipeline as a background job (50k+
    regions take minutes). Rebuilds FP sub-centroids, auto-pulls tight FP
    matches into the FP bucket, then re-partitions the good regions (the
    re-partition is skipped if a manual refine is still fresh, unless
    ``force_repartition``). Poll GET {api_prefix}/regions/cluster/status."""
    from src.services.curation.clustering.region_cluster_jobs import start_region_cluster_job

    await _ensure_indexes(opensearch)
    return await start_region_cluster_job(
        opensearch,
        max_rank=max_rank,
        auto_fp_threshold=auto_fp_threshold,
        rebuild_fp_centroids=rebuild_fp_centroids,
        force_repartition=force_repartition,
    )


@router.get('/regions/cluster/status')
async def region_cluster_status() -> dict[str, Any]:
    """Status of the background region-clustering job."""
    from src.services.curation.clustering.region_cluster_jobs import region_cluster_job_status

    return region_cluster_job_status()


@router.post('/regions/clusters/refine/{cluster_id}', responses=REGION_PROFILE_RESPONSES)
async def refine_region_cluster_endpoint(
    cluster_id: int,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
) -> dict[str, Any]:
    """Per-bucket AHC refine over the bucket's boxes; writes each box's
    ``cluster_subid`` so outliers split out."""
    from src.services.curation.clustering.region_box_clustering import refine_region_cluster
    from src.services.curation.clustering.region_cluster_jobs import mark_region_refine

    await _ensure_indexes(opensearch)
    try:
        result = await refine_region_cluster(opensearch, cluster_id)
    except Exception as exc:
        logger.error('region_refine_failed', cluster_id=cluster_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f'region refine failed: {exc}') from exc
    # Anchor the re-partition TTL on good-bucket refines so a later one-click
    # recluster won't wipe this fresh sub-cluster work (the FP bucket is rebuilt
    # by its own centroid job, not protected here).
    if cluster_id != FALSE_POSITIVE_REGION_CLUSTER_ID:
        mark_region_refine(cluster_id)
    return result


@router.get('/regions/clusters', responses=REGION_PROFILE_RESPONSES)
async def list_region_clusters(
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
    max_clusters: int = Query(500, ge=1, le=2000),
    per_cluster: int = Query(4, ge=1, le=20),
    max_rank: int | None = Query(None, ge=1),
) -> dict[str, Any]:
    """Cluster cards for the region buckets (mirrors /curation/clusters' shape).

    Clusters hold *boxes*: a card's ``size`` is the items with at least one
    box in the cluster, ``box_count`` the boxes (rows) in it -- an item with
    two boxes in one cluster is ``size`` 1, ``box_count`` 2. Most cards are
    ``candidate`` (regions are one class); the permanent false-positive
    bucket is tagged ``cluster_kind='false_positive'`` and pinned first.
    ``representatives`` are rows for the boxes closest to the centroid
    (full wire item + ``region_box_id``), shown as region close-up
    thumbnails.
    """
    F = get_region_fields()
    await _ensure_indexes(opensearch)
    # A box sits in a cluster when it is accepted or false_positive (the FP
    # bucket keeps its geometry for FP analysis/training) and carries a
    # cluster id; the same nested scope drives the aggregation.
    scope = {
        'bool': {
            'filter': [
                box_state_clause(['accepted', RegionStatus.FALSE_POSITIVE.value], F),
                {'exists': {'field': f'{F.boxes}.cluster_id'}},
            ]
        }
    }
    filters: list[dict[str, Any]] = [{'nested': {'path': F.boxes, 'query': scope}}]
    if max_rank is not None:
        filters.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})
    body = {
        'size': 0,
        'query': {'bool': {'filter': filters, 'must_not': [{'term': {'test_holdout': True}}]}},
        'aggs': {
            'boxes': {
                'nested': {'path': F.boxes},
                'aggs': {
                    'in_scope': {
                        'filter': scope,
                        'aggs': {
                            'clusters': {
                                'terms': {'field': f'{F.boxes}.cluster_id', 'size': max_clusters},
                                'aggs': {
                                    'reps': {
                                        'top_hits': {
                                            'size': per_cluster,
                                            # Only the parent _id and the box id are read.
                                            '_source': False,
                                            'docvalue_fields': [f'{F.boxes}.box_id'],
                                            'sort': [
                                                {
                                                    f'{F.boxes}.cluster_distance': {
                                                        'order': 'asc',
                                                        'missing': '_last',
                                                        'unmapped_type': 'float',
                                                    }
                                                }
                                            ],
                                        }
                                    },
                                    'subids': {
                                        'cardinality': {'field': f'{F.boxes}.cluster_subid'}
                                    },
                                    'items': {
                                        'reverse_nested': {},
                                        'aggs': {
                                            'validated': {'filter': {'term': {F.validated: True}}}
                                        },
                                    },
                                },
                            }
                        },
                    }
                },
            }
        },
    }
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    scoped = ((resp.get('aggregations') or {}).get('boxes') or {}).get('in_scope') or {}
    buckets = (scoped.get('clusters') or {}).get('buckets') or []
    clusters: list[dict[str, Any]] = []
    rep_pairs: list[tuple[str, str | None, dict[str, Any]]] = []
    for b in buckets:
        rep_hits = (((b.get('reps') or {}).get('hits') or {}).get('hits')) or []
        reps = [
            (h.get('_id') or '', ((h.get('fields') or {}).get(f'{F.boxes}.box_id') or [None])[0])
            for h in rep_hits
        ]
        rep_pairs.extend((crop_id, box_id, {}) for crop_id, box_id in reps)
        n_sub = int((b.get('subids') or {}).get('value', 0))
        is_fp = int(b['key']) == FALSE_POSITIVE_REGION_CLUSTER_ID
        items = b.get('items') or {}
        clusters.append(
            {
                'id': int(b['key']),
                'cluster_kind': RegionStatus.FALSE_POSITIVE if is_fp else 'candidate',
                'size': int(items.get('doc_count', 0)),
                'box_count': int(b['doc_count']),
                'validated_count': int((items.get('validated') or {}).get('doc_count', 0)),
                'dominant_class_id': None,
                'dominant_class_name': RegionStatus.FALSE_POSITIVE
                if is_fp
                else REGION_CLUSTER_CLASS_NAME,
                'dominant_pct': None,
                'purity': None,
                'is_unlabeled': True,
                'representative_crop_ids': [crop_id for crop_id, _box_id in reps],
                'representative_box_ids': [box_id for _crop_id, box_id in reps],
                'representative_thumb_urls': [
                    box_thumbnail_url(project_api_base(), crop_id, box_id or '')
                    for crop_id, box_id in reps
                ],
                'has_subclusters': n_sub > 0,
                'n_subclusters': n_sub,
                'updated_at': None,
            }
        )
    rows = await rows_for_pairs(
        opensearch,
        index=items_index(),
        pairs=rep_pairs,
        source_excludes=_REGION_SOURCE_EXCLUDES,
    )
    by_key = {(r['crop_id'], r['region_box_id']): r for r in rows}
    for c in clusters:
        c['representatives'] = [
            by_key[key]
            for key in zip(c['representative_crop_ids'], c['representative_box_ids'], strict=True)
            if key in by_key
        ]
    # Pin the permanent FP card first, then largest buckets.
    clusters.sort(key=lambda c: (c['cluster_kind'] != RegionStatus.FALSE_POSITIVE, -c['size']))
    return {'clusters': clusters, 'count': len(clusters)}


@router.post('/regions/fp_centroids/build', responses=REGION_PROFILE_RESPONSES)
async def build_fp_centroids_endpoint(
    opensearch: OpenSearchDep, _profile: RegionProfileDep
) -> dict[str, Any]:
    """Sub-type the FP bucket + (re)build the FP centroid store (background).

    This is the retrain trigger for the FP matcher: re-run it after marking a
    new batch of false positives so the per-sub-type centroids reflect them.
    """
    from src.services.curation.clustering.region_cluster_jobs import start_region_fp_centroid_job

    await _ensure_indexes(opensearch)
    return await start_region_fp_centroid_job(opensearch)


@router.get('/regions/fp_centroids/status')
async def fp_centroids_status() -> dict[str, Any]:
    """Background FP-centroid build job snapshot + persisted centroid metadata."""
    from src.services.curation.clustering.region_cluster_jobs import region_fp_centroid_job_status

    return region_fp_centroid_job_status()


@router.get(
    '/regions/suspected_false_positives',
    response_model=None,
    responses={**REGION_PROFILE_RESPONSES, 200: {'model': RegionRowPage}},
)
async def suspected_false_positives(
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
    threshold: float | None = Query(
        None, ge=0.0, le=2.0, description=f'Max distance; default {SUSPECTED_FP_MAX_DISTANCE}.'
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Rank non-FP region **boxes** by similarity to the known FP **sub-centroids**.

    Each row is one candidate box (``region_box_id``), matched against every
    FP sub-type centroid and scored by its nearest one (``nearest_fp_subid``),
    so a box that resembles only one flavour of false positive (e.g. a
    bumper but not a sticker) is still caught. Assists auto-labeling: an
    operator reviews the nearest matches and flips them with ``POST
    {api_prefix}/regions/batch_box_state`` (``state: false_positive``, which
    parks each box in the permanent FP bucket). Requires
    :func:`build_fp_centroids_endpoint` to have run; otherwise returns an
    empty result with ``centroids_built=false``. Rows are scored in Python,
    so this route pages rows directly: ``total == total_rows`` and
    ``page_size`` counts rows.
    """
    import time

    from src.services.curation.clustering.region_box_clustering import fp_candidate_rows
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    if threshold is None:
        threshold = SUSPECTED_FP_MAX_DISTANCE
    await _ensure_indexes(opensearch)
    store = FalsePositiveCentroidStore()
    if not store.load():
        return {
            'items': [],
            'total': 0,
            'total_rows': 0,
            'rows_truncated': False,
            'page': page,
            'page_size': page_size,
            'centroids_built': False,
            'threshold': threshold,
            'default_threshold': SUSPECTED_FP_MAX_DISTANCE,
            'message': (
                'No FP centroids yet; POST '
                # Kept as its own literal so the route-parity guard
                # (tests/integration/test_labeler_route_parity.py) can resolve
                # the path it advertises.
                f'{project_api_base()}/regions/fp_centroids/build'
                ' first.'
            ),
        }

    # The FP store is per project, and two stores may share a trained_at.
    cache_key = (current_project().record.slug, store.metadata.get('trained_at'), threshold)
    cached = _suspected_fp_cache.get(cache_key)
    now_ts = time.monotonic()
    if cached is not None and (now_ts - cached[0]) < _SUSPECTED_FP_CACHE_TTL_SEC:
        scored = cached[1]
    else:
        # Same candidate pool as the auto-pull (``fp_candidate_rows``): every
        # accepted box except those in test-holdout / human-decided items or
        # owned by a human or an import. VLM-validated boxes are included
        # (their distance to the FP centroids decides) -- real regions sit far
        # away and never surface.
        try:
            rows = await fp_candidate_rows(opensearch)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
        subids = store.metadata.get('subids', [])
        scored = []
        if rows:
            embs = np.asarray([r.vector for r in rows], dtype=np.float32)
            embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            dist, idx = store.search(embs)
            for row, d, ci in zip(rows, dist, idx, strict=True):
                if float(d) <= threshold:
                    sub = subids[int(ci)] if 0 <= int(ci) < len(subids) else None
                    scored.append((float(d), row.crop_id, row.box_id, sub))
        scored.sort(key=lambda t: t[0])
        _suspected_fp_cache[cache_key] = (now_ts, scored)
    total = len(scored)
    page_slice = scored[(page - 1) * page_size : (page - 1) * page_size + page_size]
    items = await rows_for_pairs(
        opensearch,
        index=items_index(),
        pairs=[
            (crop_id, box_id, {'suspected_fp_distance': d, 'nearest_fp_subid': sub})
            for d, crop_id, box_id, sub in page_slice
        ],
        source_excludes=_REGION_SOURCE_EXCLUDES,
    )
    return {
        'items': items,
        'total': total,
        'total_rows': total,
        'rows_truncated': False,
        'page': page,
        'page_size': page_size,
        'threshold': threshold,
        'default_threshold': SUSPECTED_FP_MAX_DISTANCE,
        'centroids_built': True,
        'trained_at': store.metadata.get('trained_at'),
    }
