"""Curation router sub-module — ingest status + lookup helpers.

The reference implementation this was ported from also carries
``POST /kb/ingest/image``, ``POST /kb/ingest/batch``,
``POST /kb/ingest/import_labels`` and ``POST /kb/ingest/import_labels_batch``
handlers backed by a domain-specific ingest service / label-import
module pair (plan §1's Bucket B — proprietary dataset-family logic,
never ported anywhere in this plan). Those four
endpoints are intentionally NOT ported here: their only real
implementation lives in code this plan declines to extract (plan §7 R5
— the generic curation stack ships with a thinner ingest path than the
reference by design, tracked as the most likely first follow-up after
merge). What *is* generic — status/backlog introspection and a path
existence lookup, both pure OpenSearch queries with no Bucket-B
dependency — is ported below unchanged.
"""

from __future__ import annotations

from typing import Any

from src.config.region_fields import get_region_fields
from src.config.region_state import RegionStatus
from src.routers.curation._common import (
    CURATION_IMAGES_INDEX,
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    _ensure_indexes,
    _PathLookupRequest,
    _PathLookupResponse,
    logger,
    router,
)


@router.get('/ingest/status')
async def ingest_status(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Recent ingest summary — counts grouped by hdd_source."""
    await _ensure_indexes(opensearch)
    body = {
        'size': 0,
        'aggs': {
            'by_source': {
                'terms': {'field': 'hdd_source', 'size': 64},
            },
            'by_day': {
                'date_histogram': {
                    'field': 'indexed_at',
                    'calendar_interval': 'day',
                    'order': {'_key': 'desc'},
                },
            },
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_IMAGES_INDEX, body=body)
    except Exception as exc:
        logger.warning('ingest_status_failed', error=str(exc))
        return {'total': 0, 'by_source': [], 'by_day': []}
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    aggs = resp.get('aggregations') or {}
    return {
        'total': total,
        'by_source': (aggs.get('by_source') or {}).get('buckets', []),
        'by_day': (aggs.get('by_day') or {}).get('buckets', [])[:14],
    }


@router.get('/ingest/sam_drain')
async def ingest_sam_drain(opensearch: OpenSearchDep) -> dict[str, int]:
    """Region-detection worklog: how many items are still waiting for the
    detection worker.

    Used by an ingest walker to decide when the asynchronous
    detect-then-verify chain has caught up after a folder finishes, before
    triggering ``/curation/pipeline/auto_label``. The walker polls this
    endpoint every ~10s and proceeds when ``pending`` reaches 0 (with a
    stability window).

    Returns a dict with five keys (transitional legacy-name rollup):

    * ``pending_detection``    — items the detection worker hasn't reached
                                  yet (region status == 'pending_detection').
                                  Legacy ``'pending'`` rows roll up here
                                  too during the migration window.
    * ``pending_verification`` — items where the detector found a
                                  candidate and the verify step is queued.
                                  Legacy ``'pending_verify'`` rows included.
    * ``pending``              — legacy alias, sum of any rows still on
                                  the old short name (transitional).
    * ``pending_verify``       — legacy alias for the same reason.
    * ``total_unfinished``     — sum of all four; what the walker polls.
    """
    await _ensure_indexes(opensearch)
    fields = get_region_fields()
    body = {
        'size': 0,
        'aggs': {
            'by_status': {
                'terms': {'field': f'{fields.status}.keyword', 'size': 16},
            },
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        logger.warning('ingest_sam_drain_failed', error=str(exc))
        return {
            'pending': 0,
            'pending_detection': 0,
            'pending_verify': 0,
            'pending_verification': 0,
            'total_unfinished': 0,
        }
    raw: dict[str, int] = {}
    for bucket in (resp.get('aggregations') or {}).get('by_status', {}).get('buckets', []):
        raw[bucket.get('key', '')] = int(bucket.get('doc_count', 0))
    pending_legacy = raw.get('pending', 0)
    pending_new = raw.get(RegionStatus.PENDING_DETECTION, 0)
    verify_legacy = raw.get('pending_verify', 0)
    verify_new = raw.get(RegionStatus.PENDING_VERIFICATION, 0)
    return {
        'pending': pending_legacy,
        'pending_detection': pending_new + pending_legacy,
        'pending_verify': verify_legacy,
        'pending_verification': verify_new + verify_legacy,
        'total_unfinished': pending_legacy + pending_new + verify_legacy + verify_new,
    }


@router.post(
    '/ingest/path_lookup',
    response_model=_PathLookupResponse,
    summary='Bulk-check which image paths are already indexed.',
)
async def curation_ingest_path_lookup(
    body: _PathLookupRequest,
    opensearch: OpenSearchDep,
) -> _PathLookupResponse:
    """Filter a list of paths down to those already ingested.

    Used by an ingest walker to short-circuit the read+hash work for
    re-scans of immutable archive media. Safe because ingest writes
    image_path verbatim and image_path is mapped as keyword on the
    images index.
    """
    if not body.image_paths:
        return _PathLookupResponse(known_paths={})

    # OpenSearch terms-query has a default 65,536 limit per call; we
    # accept up to 10k client-side and chunk internally as a margin.
    result: dict[str, str] = {}
    chunk_size = 10_000
    for i in range(0, len(body.image_paths), chunk_size):
        chunk = body.image_paths[i : i + chunk_size]
        resp = await opensearch.search(
            index=CURATION_IMAGES_INDEX,
            body={
                'size': len(chunk),
                '_source': ['image_id', 'image_path'],
                'query': {'terms': {'image_path': chunk}},
            },
        )
        for hit in resp.get('hits', {}).get('hits', []):
            src = hit.get('_source') or {}
            p = src.get('image_path')
            iid = src.get('image_id')
            if p and iid:
                result[p] = iid
    return _PathLookupResponse(known_paths=result)
