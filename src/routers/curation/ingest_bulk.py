"""``/ingest/bulk*``: bulk ingest of a folder or manifest as a background run.

A thin HTTP adapter over :mod:`src.services.curation.bulk_ingest`; the
``scripts/curation/bulk_ingest.py`` CLI and Cropwright both drive these routes, so
there is one runner. Files are read inside the API container, so a source path
must lie under a configured source root.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import Query

from src.config import get_curation_config
from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves route annotations at runtime
from src.core.disk_space import DISK_GUARD
from src.core.exceptions import RequestRejectedError
from src.routers.curation._api_error import api_error
from src.routers.curation._common import OpenSearchDep, RegistryDep, router
from src.routers.curation._ingest_bulk_models import (
    BulkIngestConfig,
    BulkIngestFailure,
    BulkIngestFailurePage,
    BulkIngestList,
    BulkIngestRequest,
    BulkIngestRun,
)
from src.routers.curation._ingest_bulk_wire import open_run, run_wire, slug
from src.routers.curation.ingest_bulk_preview import preview_config
from src.routers.curation.ingest_sources import resolve_bulk_source, start_run_now
from src.services.curation import image_serving
from src.services.curation.bulk_ingest import limits
from src.services.curation.bulk_ingest.runs import failures as run_failures, list_runs
from src.services.curation.bulk_ingest.source import (
    BulkSource,
    InvalidSourceError,
    normalise_extensions,
    validate_source,
)
from src.services.curation.bulk_ingest.sources import IngestSource, one_shot_source
from src.services.jobs.service import cancel_job


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
        preview=preview_config(),
    )


@router.post(
    '/ingest/bulk', dependencies=[DISK_GUARD], response_model=BulkIngestRun, status_code=202
)
async def start_bulk_ingest(
    body: BulkIngestRequest,
    session: DbSessionDep,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> BulkIngestRun:
    """Start a run over a folder, manifest, S3 bucket/prefix or registered source, or
    resume the interrupted, cancelled or partly failed run of the same source
    (``resumed: true``). One run per project at a time (409 ``bulk_ingest_busy``)."""
    if body.s3 is not None or body.source_id is not None:
        return await start_run_now(session, await resolve_bulk_source(body), opensearch, registry)
    kind: Literal['folder', 'manifest'] = 'folder' if body.folder is not None else 'manifest'
    location = body.folder or body.manifest or ''
    extensions = normalise_extensions(body.extensions)
    try:
        validate_source(BulkSource(kind, location, extensions))
    except InvalidSourceError as exc:
        raise api_error(422, 'bulk_ingest_invalid_source', str(exc), project=slug()) from None
    source = await one_shot_source(
        IngestSource(
            name='',
            backend=kind,
            path=location,
            extensions=list(extensions),
            source_tag=body.source,
            batch_size=body.batch_size,
            concurrency=body.concurrency,
        )
    )
    return await start_run_now(session, source, opensearch, registry, checked=True)


@router.get('/ingest/bulk', response_model=BulkIngestList)
async def list_bulk_ingests(
    session: DbSessionDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 20,
) -> BulkIngestList:
    runs = await list_runs(session, current_project_id())
    chunk = runs[(page - 1) * page_size : page * page_size]
    return BulkIngestList(
        items=[run_wire(r) for r in chunk], total=len(runs), page=page, page_size=page_size
    )


@router.get('/ingest/bulk/{run_id}', response_model=BulkIngestRun)
async def get_bulk_ingest(run_id: str, session: DbSessionDep) -> BulkIngestRun:
    return run_wire(await open_run(session, current_project_id(), run_id))


@router.get('/ingest/bulk/{run_id}/failures', response_model=BulkIngestFailurePage)
async def get_bulk_ingest_failures(
    run_id: str,
    session: DbSessionDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> BulkIngestFailurePage:
    """The failed images of the run's latest attempt, with the reason for each."""
    project = current_project_id()
    run = await open_run(session, project, run_id)
    rows, total = await run_failures(session, project, run, offset=offset, limit=limit)
    return BulkIngestFailurePage(
        items=[BulkIngestFailure(**row) for row in rows], total=total, offset=offset, limit=limit
    )


@router.post('/ingest/bulk/{run_id}/cancel', response_model=BulkIngestRun)
async def cancel_bulk_ingest(run_id: str, session: DbSessionDep) -> BulkIngestRun:
    """Stop after the batches in flight; images already stored stay stored and a
    later start of the same source continues from them."""
    project = current_project_id()
    run = await open_run(session, project, run_id)
    try:
        await cancel_job(session, uuid.UUID(run.id), project, requested_by='api')
    except RequestRejectedError as exc:
        if exc.detail.get('error') != 'job_finished':
            raise  # a run that finished keeps its status
    return run_wire(await open_run(session, project, run_id))
