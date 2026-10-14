"""``/ingest/bulk*``: bulk ingest of a folder or manifest as a background run.

A thin HTTP adapter over :mod:`src.services.curation.bulk_ingest`; the
``scripts/curation/bulk_ingest.py`` CLI and Cropwright both drive these routes, so
there is one runner. Files are read inside the API container, so a source path
must lie under a configured source root.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Query, Request

from src.config import get_curation_config
from src.routers.curation._common import OpenSearchDep, RegistryDep, _ensure_indexes, router
from src.routers.curation._config_common_models import api_error
from src.routers.curation._ingest_bulk_models import (
    BulkIngestConfig,
    BulkIngestFailure,
    BulkIngestFailurePage,
    BulkIngestList,
    BulkIngestRequest,
    BulkIngestRun,
)
from src.routers.curation._ingest_bulk_wire import open_run, own_address, run_wire, slug
from src.routers.curation.ingest import _get_ingest_service
from src.routers.curation.ingest_sources import resolve_bulk_source, start_run_now
from src.services.curation import image_serving
from src.services.curation.bulk_ingest import limits, runner, sender
from src.services.curation.bulk_ingest.source import (
    BulkSource,
    InvalidSourceError,
    normalise_extensions,
    validate_source,
)
from src.services.curation.bulk_ingest.store import list_stores


@router.get('/ingest/bulk/config', response_model=BulkIngestConfig)
async def bulk_ingest_config() -> BulkIngestConfig:
    """The roots a source must lie under and the run-option defaults and bounds."""
    return BulkIngestConfig(
        source_roots=[str(r) for r in image_serving._configured_roots(get_curation_config())],
        default_extensions=list(limits.DEFAULT_EXTENSIONS),
        default_batch_size=limits.DEFAULT_BATCH_SIZE,
        default_concurrency=limits.DEFAULT_CONCURRENCY,
        max_batch_size=limits.MAX_BATCH_SIZE,
        max_concurrency=limits.MAX_CONCURRENCY,
    )


@router.post('/ingest/bulk', response_model=BulkIngestRun, status_code=202)
async def start_bulk_ingest(
    body: BulkIngestRequest,
    request: Request,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> BulkIngestRun:
    """Start a run over a folder, manifest, S3 bucket/prefix or registered source, or
    resume the interrupted, cancelled or partly failed run of the same source
    (``resumed: true``). One run per project at a time (409 ``bulk_ingest_busy``)."""
    if body.s3 is not None or body.source_id is not None:
        return await start_run_now(resolve_bulk_source(body), request, opensearch, registry)
    source = BulkSource(
        'folder' if body.folder is not None else 'manifest',
        body.folder or body.manifest or '',
        normalise_extensions(body.extensions),
    )
    try:
        validate_source(source)
    except InvalidSourceError as exc:
        raise api_error(422, 'bulk_ingest_invalid_source', str(exc), project=slug()) from None
    await _ensure_indexes(opensearch)
    # Refuse up front (503) what the first batch would refuse: no detector, no encoder.
    await _get_ingest_service(opensearch, registry)
    address = own_address(request)
    options = runner.BulkOptions(body.source, body.batch_size, body.concurrency)
    try:
        store, resumed = runner.claim_run(source, options)
    except runner.BulkBusyError as exc:
        raise api_error(
            409, 'bulk_ingest_busy', str(exc), run_id=exc.run_id, project=slug()
        ) from None
    batch_sender = sender.loopback_sender(
        address, api_prefix=get_curation_config().api_prefix, project=slug()
    )
    runner.spawn(sender=batch_sender, store=store, source=source, options=options)
    return run_wire(store, resumed=resumed)


@router.get('/ingest/bulk', response_model=BulkIngestList)
async def list_bulk_ingests(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 20,
) -> BulkIngestList:
    stores = list_stores()
    chunk = stores[(page - 1) * page_size : page * page_size]
    return BulkIngestList(
        items=[run_wire(s) for s in chunk], total=len(stores), page=page, page_size=page_size
    )


@router.get('/ingest/bulk/{run_id}', response_model=BulkIngestRun)
async def get_bulk_ingest(run_id: str) -> BulkIngestRun:
    return run_wire(open_run(run_id))


@router.get('/ingest/bulk/{run_id}/failures', response_model=BulkIngestFailurePage)
async def get_bulk_ingest_failures(
    run_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> BulkIngestFailurePage:
    """The failed images of the run's latest attempt, with the reason for each."""
    rows, total = open_run(run_id).failures(offset=offset, limit=limit)
    return BulkIngestFailurePage(
        items=[BulkIngestFailure(**row) for row in rows], total=total, offset=offset, limit=limit
    )


@router.post('/ingest/bulk/{run_id}/cancel', response_model=BulkIngestRun)
async def cancel_bulk_ingest(run_id: str) -> BulkIngestRun:
    """Stop after the batches in flight; images already stored stay stored and a
    later start of the same source continues from them."""
    store = open_run(run_id)
    runner.cancel_run(store)
    return run_wire(store)
