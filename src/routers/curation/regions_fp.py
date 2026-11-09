"""Curation region cluster cards and the false-positive endpoints.

Split out of ``regions.py`` to keep each router sub-module focused (and
under the 700-LOC gate). Covers the region cluster cards of the serving region-box partition,
the permanent false-positive state, and the FP-centroid matcher.

False positives are heterogeneous (background clutter, similar-looking
non-target objects, empty/spurious detections), so the matcher is
**double-layered**:
:func:`~src.services.clustering.region_fp.build_region_fp_centroids`
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
from fastapi import Query

from src.config import get_region_fields
from src.config.project_context import current_project, current_project_id, project_api_base
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers._errors import server_error
from src.routers.curation._common import OpenSearchDep, RegionProfileDep, items_index, router
from src.routers.curation._error_models import REGION_PROFILE_RESPONSES
from src.routers.curation._region_row_models import RegionRowPage
from src.routers.curation.regions import region_source_excludes
from src.schemas.curation.region_fp import RegionFpCentroidsStarted
from src.services.curation.clustering.pool_snapshot import GOOD_BOX_STATES
from src.services.curation.clustering.region_box_rows import box_state_clause
from src.services.curation.region_rows import rows_for_pairs
from src.services.curation.wire import box_thumbnail_url
from src.services.jobs.service import start_job


REGION_FP_KIND = 'region_fp_centroids'


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


@router.get('/regions/clusters', responses=REGION_PROFILE_RESPONSES)
async def list_region_clusters(
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
    max_clusters: int = Query(500, ge=1, le=2000),
    per_cluster: int = Query(4, ge=1, le=20),
    max_rank: int | None = Query(None, ge=1),
) -> dict[str, Any]:
    """Cluster cards for the serving region-box partition.

    Clusters hold *boxes*: a card's ``size`` is the items with at least one
    box in the cluster, ``box_count`` the boxes (rows) in it -- an item with
    two boxes in one cluster is ``size`` 1, ``box_count`` 2. ``cluster_uid`` names the
    cluster across partitions. ``representatives`` are rows for the boxes with the highest
    ``cluster_score`` (cosine to the cluster centre; full wire item + ``region_box_id``),
    shown as region close-up thumbnails. False positives are a box state, not a card: list them
    with ``GET /regions?box_state=false_positive``.
    """
    F = get_region_fields()
    uid = f'{F.boxes}.cluster_uid'
    scope = {
        'bool': {
            'filter': [
                box_state_clause(list(GOOD_BOX_STATES), F),
                {'exists': {'field': uid}},
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
                                'terms': {'field': uid, 'size': max_clusters},
                                'aggs': {
                                    'reps': {
                                        'top_hits': {
                                            'size': per_cluster,
                                            # Only the parent _id and the box id are read.
                                            '_source': False,
                                            'docvalue_fields': [f'{F.boxes}.box_id'],
                                            'sort': [
                                                {
                                                    f'{F.boxes}.cluster_score': {
                                                        'order': 'desc',
                                                        'missing': '_last',
                                                        'unmapped_type': 'float',
                                                    }
                                                }
                                            ],
                                        }
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
        raise server_error(503, 'opensearch unavailable', exc) from exc

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
        items = b.get('items') or {}
        clusters.append(
            {
                'cluster_uid': str(b['key']),
                'size': int(items.get('doc_count', 0)),
                'box_count': int(b['doc_count']),
                'validated_count': int((items.get('validated') or {}).get('doc_count', 0)),
                'representative_crop_ids': [crop_id for crop_id, _box_id in reps],
                'representative_box_ids': [box_id for _crop_id, box_id in reps],
                'representative_thumb_urls': [
                    box_thumbnail_url(project_api_base(), crop_id, box_id or '')
                    for crop_id, box_id in reps
                ],
            }
        )
    rows = await rows_for_pairs(
        opensearch,
        index=items_index(),
        pairs=rep_pairs,
        source_excludes=await region_source_excludes(),
    )
    by_key = {(r['crop_id'], r['region_box_id']): r for r in rows}
    for c in clusters:
        c['representatives'] = [
            by_key[key]
            for key in zip(c['representative_crop_ids'], c['representative_box_ids'], strict=True)
            if key in by_key
        ]
    clusters.sort(key=lambda c: -c['size'])
    return {'clusters': clusters, 'count': len(clusters)}


@router.post(
    '/regions/fp_centroids/build',
    response_model=RegionFpCentroidsStarted,
    status_code=202,
    responses=REGION_PROFILE_RESPONSES,
)
async def build_fp_centroids_endpoint(
    session: DbSessionDep,
    _profile: RegionProfileDep,
    auto_fp_threshold: float = Query(
        0.0,
        ge=0.0,
        le=1.0,
        description='After the build, move regions within this L2 distance of an FP '
        'sub-centroid into the false_positive state (tight, near-certain matches). 0 disables.',
    ),
) -> RegionFpCentroidsStarted:
    """Sub-type the false-positive boxes and (re)build the FP centroid store, as a job.

    The retrain trigger for the FP matcher: re-run it after marking a new batch of false
    positives so the per-sub-type centroids reflect them. 202 with the job id (409 ``job_busy``
    while one is active); progress and the result are on ``GET .../jobs/{job_id}``.
    """
    started = await start_job(
        session,
        REGION_FP_KIND,
        current_project_id(),
        params={'auto_fp_threshold': auto_fp_threshold},
        requested_by='curation-api',
    )
    return RegionFpCentroidsStarted(job_id=str(started.job.id), kind=REGION_FP_KIND)


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

    from src.services.clustering.region_fp import fp_candidate_rows
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    if threshold is None:
        threshold = SUSPECTED_FP_MAX_DISTANCE
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
            raise server_error(503, 'opensearch unavailable', exc) from exc
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
        source_excludes=await region_source_excludes(),
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
