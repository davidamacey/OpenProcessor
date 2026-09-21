"""Curation region-clustering + false-positive endpoints.

Ported from the reference ``legacy_plates_fp.py`` (310 LOC). Split out of
``regions.py`` to keep each router sub-module focused (and under the
700-LOC gate). Covers the region-cluster *view* (coarse KMeans buckets
+ AHC refine), the permanent false-positive bucket, and the
FP-centroid matcher.

False positives are heterogeneous (lights, bumpers, stickers, fake
regions, empty brackets), so the matcher is **double-layered**:
:func:`~src.services.curation.clustering.orchestrator.build_region_fp_centroids`
sub-types the FP bucket into ``k`` sub-clusters and persists **one
centroid per sub-type**; :func:`suspected_false_positives` matches
each incoming region vector against *all* sub-centroids and keeps the
nearest. A new FP that doesn't resemble the bucket average is still
caught by whichever sub-type it's closest to. Re-run the build after
marking a batch of FPs to retrain the sub-centroids.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    _ensure_indexes,
    config,
    logger,
    router,
)
from src.routers.curation.regions import _REGION_SOURCE_EXCLUDES, _region_item


@router.post('/plates/cluster')
async def cluster_plates(
    opensearch: OpenSearchDep,
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
    ``force_repartition``). Poll GET /legacy/plates/cluster/status."""
    from src.services.curation.clustering.orchestrator import start_region_cluster_job

    await _ensure_indexes(opensearch)
    return await start_region_cluster_job(
        opensearch,
        max_rank=max_rank,
        auto_fp_threshold=auto_fp_threshold,
        rebuild_fp_centroids=rebuild_fp_centroids,
        force_repartition=force_repartition,
    )


@router.get('/plates/cluster/status')
async def plate_cluster_status() -> dict[str, Any]:
    """Status of the background region-clustering job."""
    from src.services.curation.clustering.orchestrator import region_cluster_job_status

    return region_cluster_job_status()


@router.post('/plates/clusters/refine/{cluster_id}')
async def refine_plate_cluster_endpoint(
    cluster_id: int,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Per-bucket AHC refine; writes RegionFields.cluster_subid so outliers split out."""
    from src.services.curation.clustering.orchestrator import (
        FALSE_POSITIVE_REGION_CLUSTER_ID,
        mark_region_refine,
        refine_region_cluster,
    )

    await _ensure_indexes(opensearch)
    try:
        result = await refine_region_cluster(opensearch, cluster_id)
    except Exception as exc:
        logger.error('legacy_plate_refine_failed', cluster_id=cluster_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f'plate refine failed: {exc}') from exc
    # Anchor the re-partition TTL on good-bucket refines so a later one-click
    # recluster won't wipe this fresh sub-cluster work (the FP bucket is rebuilt
    # by its own centroid job, not protected here).
    if cluster_id != FALSE_POSITIVE_REGION_CLUSTER_ID:
        mark_region_refine(cluster_id)
    return result


@router.get('/plates/clusters')
async def list_plate_clusters(
    opensearch: OpenSearchDep,
    max_clusters: int = Query(500, ge=1, le=2000),
    per_cluster: int = Query(4, ge=1, le=20),
    max_rank: int | None = Query(None, ge=1),
) -> dict[str, Any]:
    """Cluster cards for the region buckets (mirrors /legacy/clusters' shape).

    Most cards are ``candidate`` (regions are one class); the permanent
    false-positive bucket is tagged ``cluster_kind='false_positive'`` and
    pinned first. Reps are the regions closest to the centroid, shown as
    region close-up thumbnails.
    """
    from src.services.curation.clustering.orchestrator import FALSE_POSITIVE_REGION_CLUSTER_ID

    F = get_region_fields()
    await _ensure_indexes(opensearch)
    must: list[dict[str, Any]] = [
        {'exists': {'field': F.cluster_id}},
        {'exists': {'field': F.bbox_norm}},
    ]
    if max_rank is not None:
        must.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})
    body = {
        'size': 0,
        'query': {'bool': {'must': must, 'must_not': [{'term': {'test_holdout': True}}]}},
        'aggs': {
            'clusters': {
                'terms': {'field': F.cluster_id, 'size': max_clusters},
                'aggs': {
                    'reps': {
                        'top_hits': {
                            'size': per_cluster,
                            '_source': ['crop_id'],
                            'sort': [
                                {
                                    F.cluster_distance: {
                                        'order': 'asc',
                                        'missing': '_last',
                                        'unmapped_type': 'float',
                                    }
                                }
                            ],
                        }
                    },
                    'subids': {'cardinality': {'field': F.cluster_subid}},
                    'validated': {'filter': {'term': {F.validated: True}}},
                },
            }
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    buckets = (((resp.get('aggregations') or {}).get('clusters') or {}).get('buckets')) or []
    clusters: list[dict[str, Any]] = []
    for b in buckets:
        rep_hits = (((b.get('reps') or {}).get('hits') or {}).get('hits')) or []
        rep_ids = [h.get('_id') or '' for h in rep_hits]
        n_sub = int((b.get('subids') or {}).get('value', 0))
        is_fp = int(b['key']) == FALSE_POSITIVE_REGION_CLUSTER_ID
        clusters.append(
            {
                'id': int(b['key']),
                'cluster_kind': RegionStatus.FALSE_POSITIVE if is_fp else 'candidate',
                'size': int(b['doc_count']),
                'validated_count': int((b.get('validated') or {}).get('doc_count', 0)),
                'dominant_class_id': None,
                'dominant_class_name': RegionStatus.FALSE_POSITIVE if is_fp else 'license_plate',
                'dominant_pct': None,
                'purity': None,
                'is_unlabeled': True,
                'representative_crop_ids': rep_ids,
                'representative_thumb_urls': [
                    f'{config.api_prefix}/crops/{cid}/region_thumbnail' for cid in rep_ids
                ],
                'has_subclusters': n_sub > 0,
                'n_subclusters': n_sub,
                'updated_at': None,
            }
        )
    # Pin the permanent FP card first, then largest buckets.
    clusters.sort(key=lambda c: (c['cluster_kind'] != RegionStatus.FALSE_POSITIVE, -c['size']))
    return {'clusters': clusters, 'count': len(clusters)}


@router.post('/plates/fp_centroids/build')
async def build_fp_centroids_endpoint(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Sub-type the FP bucket + (re)build the FP centroid store (background).

    This is the retrain trigger for the FP matcher: re-run it after marking a
    new batch of false positives so the per-sub-type centroids reflect them.
    """
    from src.services.curation.clustering.orchestrator import start_region_fp_centroid_job

    await _ensure_indexes(opensearch)
    return await start_region_fp_centroid_job(opensearch)


@router.get('/plates/fp_centroids/status')
async def fp_centroids_status() -> dict[str, Any]:
    """Background FP-centroid build job snapshot + persisted centroid metadata."""
    from src.services.curation.clustering.orchestrator import region_fp_centroid_job_status

    return region_fp_centroid_job_status()


@router.get('/plates/suspected_false_positives')
async def suspected_false_positives(
    opensearch: OpenSearchDep,
    threshold: float = Query(0.35, ge=0.0, le=2.0),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Rank non-FP region crops by similarity to the known FP **sub-centroids**.

    Each candidate is matched against every FP sub-type centroid and scored by
    its nearest one (``nearest_fp_subid``), so a crop that resembles only one
    flavour of false positive (e.g. a bumper but not a sticker) is still caught.
    Assists auto-labeling: an operator reviews the nearest matches and
    bulk-confirms via ``POST /legacy/plates/batch_status`` (which routes them into
    the permanent FP bucket). Requires :func:`build_fp_centroids_endpoint` to
    have run; otherwise returns an empty result with ``centroids_built=false``.
    """
    import numpy as np

    from src.services.curation.clustering.orchestrator import fp_candidate_must_not
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    F = get_region_fields()
    await _ensure_indexes(opensearch)
    store = FalsePositiveCentroidStore()
    if not store.load():
        return {
            'items': [],
            'total': 0,
            'page': page,
            'page_size': page_size,
            'centroids_built': False,
            'message': 'No FP centroids yet; POST /legacy/plates/fp_centroids/build first.',
        }

    # Same candidate pool as the auto-pull: everything except already-FP,
    # test-holdout, and HUMAN-decided crops. VLM-validated/detected crops are
    # included (their distance to the FP centroids decides) — real regions
    # sit far away and never surface, so include_detected is no longer a
    # useful gate.
    must = [{'exists': {'field': F.embedding}}]
    must_not = fp_candidate_must_not()
    subids = store.metadata.get('subids', [])
    scored: list[tuple[float, str, str | None]] = []
    body = {
        'size': 2000,
        'query': {'bool': {'must': must, 'must_not': must_not}},
        '_source': {'includes': [F.embedding]},
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body, scroll='5m')
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        embs = np.asarray(
            [(h.get('_source') or {}).get(F.embedding) for h in hits],
            dtype=np.float32,
        )
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        dist, idx = store.search(embs)
        for h, d, ci in zip(hits, dist, idx, strict=True):
            if float(d) <= threshold:
                sub = subids[int(ci)] if 0 <= int(ci) < len(subids) else None
                scored.append((float(d), h['_id'], sub))
        resp = await opensearch.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
    if scroll_id:
        try:
            await opensearch.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:
            logger.debug('legacy_suspected_fp_clear_scroll_failed', error=str(exc))

    scored.sort(key=lambda t: t[0])
    total = len(scored)
    page_slice = scored[(page - 1) * page_size : (page - 1) * page_size + page_size]
    items: list[dict[str, Any]] = []
    ids = [cid for _d, cid, _s in page_slice]
    if ids:
        docs = await opensearch.mget(
            index=CURATION_ITEMS_INDEX,
            body={'ids': ids},
            _source={'excludes': _REGION_SOURCE_EXCLUDES},
        )
        by_id = {d['_id']: (d.get('_source') or {}) for d in docs['docs'] if d.get('found')}
        for d, cid, sub in page_slice:
            item = _region_item(by_id.get(cid, {}), cid)
            item['suspected_fp_distance'] = d
            item['nearest_fp_subid'] = sub
            items.append(item)
    return {
        'items': items,
        'total': total,
        'page': page,
        'page_size': page_size,
        'threshold': threshold,
        'centroids_built': True,
        'trained_at': store.metadata.get('trained_at'),
    }
