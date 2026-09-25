"""Curation router sub-module — ingest, label import, and status/lookup helpers.

``POST /ingest/image``, ``POST /ingest/batch``, ``POST /ingest/upload``,
``POST /import_labels`` and ``POST /import_labels/batch`` are the generic
curation ingest front door: they create ``images`` + ``items`` documents (and, for label
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
from typing import Annotated, Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.config import DetectionProfile, RegionStatus, get_region_fields
from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.routers.curation._common import (
    CURATION_IMAGES_INDEX,
    CURATION_ITEMS_INDEX,
    BatchIngestResponse as _BatchIngestResponse,
    BatchIngestSummaryResponse as _BatchIngestSummaryResponse,
    ImportLabelsBatchRequest,
    ImportLabelsRequest,
    IngestBatchConfig,
    IngestConfigResponse,
    IngestImageRequest,
    IngestImageResponse,
    IngestRegionDrainConfig,
    IngestRegionDrainResponse,
    IngestStatusResponse,
    IngestUploadConfig,
    OpenSearchDep,
    RegionDependencyStatusResponse,
    RegistryDep,
    _ensure_indexes,
    _PathLookupRequest,
    _PathLookupResponse,
    router,
)
from src.services.curation.image_serving import UNSERVABLE_PATH_ERROR, is_servable_image_path
from src.services.curation.ingest import CurationIngestService
from src.services.curation.ingest_models import ERROR_KIND_DECODE_FAILED, ERROR_KIND_UNSERVABLE_PATH
from src.services.curation.label_import import (
    DEFAULT_LABEL_SOURCE,
    count_disagreements,
    import_labels_batch,
    import_yolo_labels,
)


class IngestBatchItem(IngestImageRequest):
    """One batch entry: an image, and optionally its ground-truth labels.

    ``label_txt_path`` is what makes an "ingest an already-labeled
    dataset and compare the detector against ground truth" pass a single
    call instead of an ingest followed by a second
    ``POST /import_labels/batch`` round trip.
    """

    label_txt_path: str | None = Field(
        default=None,
        description='Optional companion YOLO .txt label file for this image',
    )


class IngestBatchRequest(BaseModel):
    # F-22: extra='forbid' + a required, non-empty items list. Previously a
    # body with a wrong key (e.g. {'paths': [...]}) validated fine with
    # items defaulting to [] and the endpoint returned 200 status=success
    # with all-zero counts -- a silent no-op indistinguishable from "ingested
    # an empty batch on purpose". Both a stale key and a missing/empty
    # items list now 422 instead.
    model_config = ConfigDict(extra='forbid')

    items: list[IngestBatchItem] = Field(..., min_length=1)
    label_source: str = Field(
        default=DEFAULT_LABEL_SOURCE,
        description='label_source recorded on labels imported from label_txt_path',
    )
    detect_mismatches: bool = Field(
        default=False,
        description=(
            'Flag labels whose IoU-matched item carried a different detector class, '
            'and report the count as summary.mismatches'
        ),
    )


def _get_detection_profile() -> DetectionProfile:
    """The primary item-proposal detector profile (``OP_INGEST_PRIMARY_*``,
    see :func:`src.config.ingest_profiles.ingest_primary_profile`)."""
    return ingest_primary_profile()


def _get_secondary_profile() -> DetectionProfile | None:
    """The optional secondary classifier profile (``OP_INGEST_SECONDARY_*``)."""
    return ingest_secondary_profile()


async def _get_ingest_service(opensearch: Any, registry: Any) -> CurationIngestService:
    from src.main import app, get_async_triton_pool

    pe_encoder = getattr(app.state, 'pe_encoder', None)
    if pe_encoder is None:
        raise HTTPException(
            status_code=503,
            detail='PE encoder not initialized; ingest is unavailable until app startup completes',
        )
    try:
        profile = _get_detection_profile()
        secondary = _get_secondary_profile()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f'ingest misconfigured: {exc}') from exc
    if not profile.detector_model:
        raise HTTPException(
            status_code=503,
            detail=(
                'No detector configured for ingest — set OP_INGEST_PRIMARY_DETECTOR_MODEL '
                'to a Triton model name that serves item proposals for this deployment'
            ),
        )
    return CurationIngestService(
        opensearch=opensearch,
        triton_pool=get_async_triton_pool(),
        registry=registry,
        profile=profile,
        secondary_profile=secondary,
        pe_encoder=pe_encoder,
    )


@router.post('/ingest/image', response_model=IngestImageResponse)
async def curation_ingest_image(
    body: IngestImageRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> IngestImageResponse:
    """Ingest a single image from a path already reachable inside the container."""
    if not is_servable_image_path(body.path):
        raise HTTPException(status_code=422, detail=f'{body.path}: {UNSERVABLE_PATH_ERROR}')
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
        n_regions=result.n_region_queued,
        error=result.error,
        error_kind=result.error_kind,
        source_identifier=result.source_identifier,
        secondary_detector_error=result.secondary_detector_error,
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
    whole batch. Items that carry a ``label_txt_path`` also have their
    ground-truth YOLO labels imported in the same call.

    The whole-image detector inference is issued in batched Triton calls
    (one per ``DetectionProfile.batch_limit`` chunk), so a larger batch
    is materially faster than the same images posted one at a time.
    """
    from src.config import get_curation_config

    max_items = get_curation_config().batch_max_items_per_request
    if len(body.items) > max_items:
        raise HTTPException(
            status_code=413,
            detail=f'{len(body.items)} items exceeds the per-request limit of {max_items}',
        )
    await _ensure_indexes(opensearch)
    service = await _get_ingest_service(opensearch, registry)

    images: list[bytes] = []
    paths: list[str] = []
    label_paths: list[str | None] = []
    failed_early: list[IngestImageResponse] = []
    for item in body.items:
        if not is_servable_image_path(item.path):
            failed_early.append(
                IngestImageResponse(
                    status='failed',
                    image_path=item.path,
                    error=UNSERVABLE_PATH_ERROR,
                    error_kind=ERROR_KIND_UNSERVABLE_PATH,
                )
            )
            continue
        if item.label_txt_path is not None and not is_servable_image_path(item.label_txt_path):
            # label_txt_path gets the same root guard as the image
            # path -- a client-controlled label file path must not escape
            # the configured source roots either.
            failed_early.append(
                IngestImageResponse(
                    status='failed',
                    image_path=item.path,
                    error=f'label_txt_path {item.label_txt_path!r}: {UNSERVABLE_PATH_ERROR}',
                    error_kind=ERROR_KIND_UNSERVABLE_PATH,
                )
            )
            continue
        try:
            images.append(Path(item.path).read_bytes())
            paths.append(item.path)
            label_paths.append(item.label_txt_path)
        except OSError as exc:
            failed_early.append(
                IngestImageResponse(
                    status='failed',
                    image_path=item.path,
                    error=str(exc),
                    error_kind=ERROR_KIND_DECODE_FAILED,
                )
            )

    # The service stamps one source per call; honour the per-item tag when
    # the batch agrees on one (the common case — a driver tags a whole run).
    sources = {item.source for item in body.items}
    batch_result = (
        await service.ingest_batch(
            images,
            paths,
            label_paths=label_paths if any(label_paths) else None,
            source=sources.pop() if len(sources) == 1 else 'batch',
            label_source=body.label_source,
            detect_mismatches=body.detect_mismatches,
        )
        if images
        else None
    )
    return _batch_response(batch_result, failed_early)


def _batch_response(
    batch_result: Any,
    failed_early: list[IngestImageResponse],
) -> _BatchIngestResponse:
    """Merge per-item pre-flight failures with a service ``BatchIngestResult``."""
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
                n_regions=r.n_region_queued,
                error=r.error,
                error_kind=r.error_kind,
                source_identifier=r.source_identifier,
                secondary_detector_error=r.secondary_detector_error,
            )
            for r in batch_result.results
        )
        summary.successful += batch_result.summary.successful
        summary.duplicates += batch_result.summary.duplicates
        summary.failed += batch_result.summary.failed
        summary.crops_indexed += batch_result.summary.crops_indexed
        summary.labels_imported += batch_result.summary.labels_imported
        summary.mismatches += batch_result.summary.mismatches
        summary.missed_labels += batch_result.summary.missed_labels
        summary.unmatched_detections += batch_result.summary.unmatched_detections
        summary.secondary_detector_failures += batch_result.summary.secondary_detector_failures

    if summary.failed == 0:
        status: Any = 'success'
    elif summary.successful == 0:
        status = 'error'
    else:
        status = 'partial'
    return _BatchIngestResponse(
        status=status,
        summary=summary,
        results=results,
        disagreements=list(batch_result.disagreements) if batch_result is not None else [],
    )


@router.post('/import_labels')
async def curation_import_labels(
    body: ImportLabelsRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> dict[str, int]:
    """Import a single YOLO ``.txt`` label file against an already-ingested image."""
    await _ensure_indexes(opensearch)
    mismatches: list[dict[str, Any]] = []
    n = await import_yolo_labels(
        Path(body.image_path),
        Path(body.label_txt_path),
        registry,
        opensearch,
        label_source=body.label_source or DEFAULT_LABEL_SOURCE,
        detect_mismatches=body.detect_mismatches,
        mismatch_sink=mismatches,
    )
    return {'labels_imported': n, **count_disagreements(mismatches)}


@router.post('/import_labels/batch')
async def curation_import_labels_batch(
    body: ImportLabelsBatchRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> dict[str, Any]:
    """Batch-import YOLO ``.txt`` label files against already-ingested images.

    With ``detect_mismatches`` the response also carries the per-label
    ``disagreements`` records (same shape as ``POST /ingest/batch``).
    """
    await _ensure_indexes(opensearch)
    pairs = [(Path(i.image_path), Path(i.label_txt_path)) for i in body.items]
    label_source = body.items[0].label_source if body.items else DEFAULT_LABEL_SOURCE
    detect_mismatches = any(i.detect_mismatches for i in body.items)
    disagreements: list[dict[str, Any]] = []
    summary: dict[str, Any] = dict(
        await import_labels_batch(
            pairs,
            registry,
            opensearch,
            label_source=label_source,
            detect_mismatches=detect_mismatches,
            disagreement_sink=disagreements,
        )
    )
    summary['disagreements'] = disagreements
    return summary


@router.get('/ingest/status', response_model=IngestStatusResponse)
async def ingest_status(
    opensearch: OpenSearchDep,
    run_id: Annotated[
        str | None,
        Query(description='Scope counts to one POST /ingest/upload run_id.'),
    ] = None,
) -> IngestStatusResponse:
    """Recent ingest summary — counts grouped by source.

    ``run_id`` scopes ``total``/``by_source``/``by_day`` to images carrying
    that ``ingest_run_id`` -- an upload run's own images doc field,
    set by ``POST /ingest/upload``'s optional ``run_id`` form field.
    """
    await _ensure_indexes(opensearch)
    query: dict[str, Any] = (
        {'term': {'ingest_run_id': run_id}} if run_id is not None else {'match_all': {}}
    )
    body = {
        'size': 0,
        'track_total_hits': True,
        'query': query,
        'aggs': {
            'by_source': {
                'terms': {'field': 'source', 'size': 64},
            },
            # Filter to the last 14 days in the query rather than
            # date-histogramming the whole index and slicing to [:14] in
            # Python -- the histogram used to run over every doc ever
            # indexed just to keep the first 14 desc-sorted buckets.
            'by_day': {
                'filter': {'range': {'indexed_at': {'gte': 'now-14d/d'}}},
                'aggs': {
                    'days': {
                        'date_histogram': {
                            'field': 'indexed_at',
                            'calendar_interval': 'day',
                            'order': {'_key': 'desc'},
                        },
                    },
                },
            },
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_IMAGES_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    aggs = resp.get('aggregations') or {}
    return IngestStatusResponse(
        total=total,
        by_source=(aggs.get('by_source') or {}).get('buckets', []),
        by_day=((aggs.get('by_day') or {}).get('days') or {}).get('buckets', []),
    )


@router.get('/ingest/config', response_model=IngestConfigResponse)
async def ingest_config() -> IngestConfigResponse:
    """Typed ingest capability + limits.

    ``upload``/``batch``/``region_drain`` are all real config, not
    hardcoded client-side constants: ``upload.max_images_per_request`` /
    ``max_bytes_per_request`` / ``accepted_extensions`` are enforced (413 /
    per-item ``unsupported_type``) by ``POST /ingest/upload``;
    ``batch.max_items`` is enforced by ``POST /ingest/batch``;
    ``region_drain.*`` are the parameters ``GET /ingest/region_drain``'s
    stability verdict uses.
    """
    from src.config import get_curation_config
    from src.services.curation.image_serving import _configured_roots
    from src.services.curation.region_drain import (
        region_drain_poll_interval_s,
        region_drain_stable_polls,
    )

    cfg = get_curation_config()
    return IngestConfigResponse(
        upload=IngestUploadConfig(
            max_images_per_request=cfg.upload_max_images_per_request,
            max_bytes_per_request=cfg.upload_max_bytes_per_request,
            accepted_extensions=list(cfg.upload_accepted_extensions),
            persists_bytes=True,
        ),
        batch=IngestBatchConfig(
            max_items=cfg.batch_max_items_per_request,
            source_roots=[str(r) for r in _configured_roots(cfg)],
        ),
        region_drain=IngestRegionDrainConfig(
            poll_interval_s=region_drain_poll_interval_s(),
            stable_polls=region_drain_stable_polls(),
        ),
    )


@router.get('/ingest/region_drain', response_model=IngestRegionDrainResponse)
async def ingest_region_drain(opensearch: OpenSearchDep) -> IngestRegionDrainResponse:
    """Region-detection worklog: how many items are still waiting for the
    detection worker.

    Used by an ingest walker to decide when the asynchronous
    detect-then-verify chain has caught up after a folder finishes, before
    triggering ``/curation/pipeline/auto_label``. The walker polls this
    endpoint every ``region_drain.poll_interval_s`` (``GET /ingest/config``)
    and proceeds once ``drained`` is true -- computed server-side
    now, not invented client-side from ``total_unfinished == 0``.

    Returns:
    * ``pending_detection``    — items the detection worker hasn't reached
                                  yet (region status == 'pending_detection').
    * ``pending_verification`` — items where the detector found a
                                  candidate and the verify step is queued.
    * ``total_unfinished``     — sum of the two; what a legacy walker polled.
    * ``drained``              — true once ``total_unfinished`` has read 0
                                  for ``region_drain.stable_polls`` consecutive
                                  polls of this endpoint.
    * ``stable_for_s``         — seconds since the last non-zero reading.
    * ``observed_at``          — this poll's timestamp.
    * ``region_dependencies``  — (V-1) the active region profile's Triton
                                  model(s) (detector/segmenter) and
                                  whether each is READY right now; empty
                                  when no region profile is configured.
    * ``stall_reason``         — a human-readable line when items are
                                  pending AND a dependency is down; null
                                  otherwise (nothing pending, or the
                                  worker just hasn't caught up yet).

    Re-ingested data can never carry the retired ``'pending'`` /
    ``'pending_verify'`` short names, so there is no legacy rollup.
    """
    from src.services.curation.region_dependency_health import (
        check_region_dependencies,
        stall_reason as _stall_reason,
    )
    from src.services.curation.region_drain import observe_drain
    from src.services.detection.profile_registry import region_profile_or_neutral
    from src.services.triton_control import TritonControlService

    await _ensure_indexes(opensearch)
    fields = get_region_fields()
    body = {
        'size': 0,
        'aggs': {
            'by_status': {
                'terms': {'field': fields.status, 'size': 16},
            },
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        # Never answer zeros on an outage: total_unfinished == 0 is the
        # "worker caught up" signal a walker proceeds on.
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    raw: dict[str, int] = {}
    for bucket in (resp.get('aggregations') or {}).get('by_status', {}).get('buckets', []):
        raw[bucket.get('key', '')] = int(bucket.get('doc_count', 0))
    pending_detection = raw.get(RegionStatus.PENDING_DETECTION, 0)
    pending_verification = raw.get(RegionStatus.PENDING_VERIFICATION, 0)
    total_unfinished = pending_detection + pending_verification
    verdict = observe_drain(total_unfinished)

    control = TritonControlService()
    dependencies = await check_region_dependencies(
        control.get_repository_index, region_profile_or_neutral()
    )
    dependency_responses = [
        RegionDependencyStatusResponse(
            role=d.role,
            model=d.model,
            ready=d.ready,
            unavailable_since=d.unavailable_since,
            detail=d.detail,
        )
        for d in dependencies
    ]

    return IngestRegionDrainResponse(
        pending_detection=pending_detection,
        pending_verification=pending_verification,
        total_unfinished=total_unfinished,
        drained=verdict.drained,
        stable_for_s=verdict.stable_for_s,
        observed_at=verdict.observed_at,
        region_dependencies=dependency_responses,
        stall_reason=_stall_reason(dependencies, pending_detection=pending_detection),
    )


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
    re-scans of immutable archive media. Matches on ``image_path`` OR
    ``source_identifier``: a byte-upload ingest's client identifier
    is recorded as ``source_identifier`` now that ``image_path`` is the
    server-persisted path, so a re-scan driver that only knows its own
    identifiers still gets a hit. Both fields are mapped keyword on the
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
                '_source': ['image_id', 'image_path', 'source_identifier'],
                'query': {
                    'bool': {
                        'should': [
                            {'terms': {'image_path': chunk}},
                            {'terms': {'source_identifier': chunk}},
                        ],
                        'minimum_should_match': 1,
                    }
                },
            },
        )
        chunk_set = set(chunk)
        for hit in resp.get('hits', {}).get('hits', []):
            src = hit.get('_source') or {}
            iid = src.get('image_id')
            if not iid:
                continue
            # Key the result by whichever of the two fields is one of the
            # requested paths -- a doc found via source_identifier keys
            # under that identifier, not the (unrelated to the caller)
            # persisted image_path.
            for field in ('image_path', 'source_identifier'):
                p = src.get(field)
                if p and p in chunk_set:
                    result[p] = iid
    return _PathLookupResponse(known_paths=result)
