"""Curation router sub-module — ingest, label import, and status/lookup helpers.

``POST /ingest/image``, ``POST /ingest/batch``, ``POST /import_labels``
and ``POST /import_labels/batch`` are the generic curation ingest front
door: they create ``images`` + ``items`` documents (and, for label
import, ``labels_confirmed`` documents), backed by
:class:`~src.services.curation.ingest.CurationIngestService` and
:mod:`src.services.curation.label_import`. Everything else in this
module (status/backlog introspection, the path-existence lookup) is
unchanged pure-OpenSearch read queries.

The ingest endpoints require a detector to be configured — a
``DetectionProfile`` naming a Triton model that already serves item
proposals for this deployment. Bring your own trained detector; this
module ships no domain-specific class taxonomy or detector weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.config import DetectionProfile, get_region_fields
from src.routers.curation._common import (
    CURATION_IMAGES_INDEX,
    CURATION_ITEMS_INDEX,
    BatchIngestResponse as _BatchIngestResponse,
    BatchIngestSummaryResponse as _BatchIngestSummaryResponse,
    ImportLabelsBatchRequest,
    ImportLabelsRequest,
    IngestImageRequest,
    IngestImageResponse,
    OpenSearchDep,
    RegistryDep,
    _ensure_indexes,
    _PathLookupRequest,
    _PathLookupResponse,
    logger,
    router,
)
from src.services.curation.ingest import CurationIngestService
from src.services.curation.label_import import (
    DEFAULT_LABEL_SOURCE,
    import_labels_batch,
    import_yolo_labels,
)


class IngestBatchRequest(BaseModel):
    items: list[IngestImageRequest] = Field(default_factory=list)


def _get_detection_profile() -> DetectionProfile:
    """The primary-detector profile for the ingest pipeline.

    Env-configurable via ``OP_DETECTION_*`` (see
    ``DetectionProfile.from_env``) — a deployment brings its own
    detector by setting ``OP_DETECTION_DETECTOR_MODEL`` at minimum.
    """
    return DetectionProfile.from_env(name='item')


async def _get_ingest_service(opensearch: Any, registry: Any) -> CurationIngestService:
    from src.main import app, get_async_triton_pool

    pe_encoder = getattr(app.state, 'pe_encoder', None)
    if pe_encoder is None:
        raise HTTPException(
            status_code=503,
            detail='PE encoder not initialized; ingest is unavailable until app startup completes',
        )
    profile = _get_detection_profile()
    if not profile.detector_model:
        raise HTTPException(
            status_code=503,
            detail=(
                'No detector configured for ingest — set OP_DETECTION_DETECTOR_MODEL to a '
                'Triton model name that serves item proposals for this deployment'
            ),
        )
    return CurationIngestService(
        opensearch=opensearch,
        triton_pool=get_async_triton_pool(),
        registry=registry,
        profile=profile,
        pe_encoder=pe_encoder,
    )


@router.post('/ingest/image', response_model=IngestImageResponse)
async def curation_ingest_image(
    body: IngestImageRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> IngestImageResponse:
    """Ingest a single image from a path already reachable inside the container."""
    await _ensure_indexes(opensearch)
    path = Path(body.path)
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=404, detail=f'cannot read {body.path}: {exc}') from None

    service = await _get_ingest_service(opensearch, registry)
    result = await service.ingest_one(image_bytes, body.path, source=body.source)
    return IngestImageResponse(
        status=result.status,
        image_id=result.image_id,
        image_path=result.image_path,
        imohash=result.imohash,
        n_crops=result.n_crops,
        error=result.error,
    )


@router.post('/ingest/batch', response_model=_BatchIngestResponse)
async def curation_ingest_batch(
    body: IngestBatchRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> _BatchIngestResponse:
    """Ingest a batch of images from paths already reachable inside the container.

    Every item shares its ``source`` tag independently; a per-item read
    failure is reported as a ``failed`` result rather than aborting the
    whole batch.
    """
    await _ensure_indexes(opensearch)
    service = await _get_ingest_service(opensearch, registry)

    images: list[bytes] = []
    paths: list[str] = []
    failed_early: list[IngestImageResponse] = []
    for item in body.items:
        try:
            images.append(Path(item.path).read_bytes())
            paths.append(item.path)
        except OSError as exc:
            failed_early.append(
                IngestImageResponse(status='failed', image_path=item.path, error=str(exc))
            )

    batch_result = await service.ingest_batch(images, paths) if images else None
    results = list(failed_early)
    summary = _BatchIngestSummaryResponse(failed=len(failed_early))
    if batch_result is not None:
        results.extend(
            IngestImageResponse(
                status=r.status,
                image_id=r.image_id,
                image_path=r.image_path,
                imohash=r.imohash,
                n_crops=r.n_crops,
                error=r.error,
            )
            for r in batch_result.results
        )
        summary.successful += batch_result.summary.successful
        summary.duplicates += batch_result.summary.duplicates
        summary.failed += batch_result.summary.failed
        summary.crops_indexed += batch_result.summary.crops_indexed

    if summary.failed == 0:
        status: Any = 'success'
    elif summary.successful == 0:
        status = 'error'
    else:
        status = 'partial'
    return _BatchIngestResponse(status=status, summary=summary, results=results)


@router.post('/import_labels')
async def curation_import_labels(
    body: ImportLabelsRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> dict[str, int]:
    """Import a single YOLO ``.txt`` label file against an already-ingested image."""
    await _ensure_indexes(opensearch)
    n = await import_yolo_labels(
        Path(body.image_path),
        Path(body.label_txt_path),
        registry,
        opensearch,
        label_source=body.label_source or DEFAULT_LABEL_SOURCE,
    )
    return {'labels_imported': n}


@router.post('/import_labels/batch')
async def curation_import_labels_batch(
    body: ImportLabelsBatchRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> dict[str, int]:
    """Batch-import YOLO ``.txt`` label files against already-ingested images."""
    await _ensure_indexes(opensearch)
    pairs = [(Path(i.image_path), Path(i.label_txt_path)) for i in body.items]
    label_source = body.items[0].label_source if body.items else DEFAULT_LABEL_SOURCE
    return await import_labels_batch(pairs, registry, opensearch, label_source=label_source)


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
    pending_new = raw.get('pending_detection', 0)
    verify_legacy = raw.get('pending_verify', 0)
    verify_new = raw.get('pending_verification', 0)
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
