"""``/ingest/bulk*``: bulk ingest of a folder or manifest as a background run.

A thin HTTP adapter over :mod:`src.services.curation.bulk_ingest`; the
``scripts/curation/bulk_ingest.py`` CLI and Cropwright both drive these routes, so
there is one runner. Files are read inside the API container, so a source path
must lie under a configured source root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import HTTPException, Query, Request

from src.config import get_curation_config
from src.routers.curation._common import OpenSearchDep, RegistryDep, _ensure_indexes, router
from src.routers.curation._config_common_models import api_error
from src.routers.curation._ingest_bulk_models import (
    BulkIngestConfig,
    BulkIngestFailure,
    BulkIngestFailurePage,
    BulkIngestList,
    BulkIngestProgress,
    BulkIngestRequest,
    BulkIngestRun,
    BulkOptionsWire,
    BulkSourceWire,
)
from src.routers.curation.ingest import _get_ingest_service
from src.services.curation import image_serving
from src.services.curation.bulk_ingest import limits, runner, sender
from src.services.curation.bulk_ingest.source import (
    BulkSource,
    InvalidSourceError,
    normalise_extensions,
    validate_source,
)
from src.services.curation.bulk_ingest.store import (
    ACTIVE_STATUSES,
    RESUMABLE_STATUSES,
    BulkStore,
    list_stores,
    open_store,
)


if TYPE_CHECKING:
    from typing import Any

POLL_AFTER_S = 2.0
_PROGRESS_KEYS = {
    'discovered': 0,
    'discovery_complete': False,
    'skipped': 0,
    'ingested': 0,
    'duplicates': 0,
    'failed': 0,
    'processed': 0,
    'crops_indexed': 0,
    'n_embedded': 0,
    'n_not_embedded': 0,
    'n_embed_failed': 0,
    'in_flight': 0,
}


def _slug() -> str:
    return get_curation_config().project_slug


def _own_address(request: Request) -> tuple[str, int]:
    """The API's own listening socket, which batches are sent back to so they spread
    over every worker process."""
    server = request.scope.get('server')
    if not server or server[1] is None:
        raise HTTPException(
            status_code=503, detail='bulk ingest needs the API to listen on a TCP socket'
        )
    return str(server[0]), int(server[1])


def _open(run_id: str) -> BulkStore:
    store = open_store(run_id)
    if store is None:
        raise api_error(404, 'bulk_ingest_not_found', f'no bulk ingest {run_id!r}', project=_slug())
    return store


def run_wire(store: BulkStore, *, resumed: bool = False) -> BulkIngestRun:
    state: dict[str, Any] = store.repaired_state()
    status = state.get('status', 'queued')
    source = state.get('source') or {}
    options = state.get('options') or {}
    progress = {key: state.get(key, default) for key, default in _PROGRESS_KEYS.items()}
    return BulkIngestRun(
        project=_slug(),
        run_id=store.run_id,
        status=status,
        resumed=resumed,
        source=BulkSourceWire(
            kind=source.get('kind', 'folder'),
            path=source.get('path', ''),
            extensions=list(source.get('extensions') or []),
        ),
        options=BulkOptionsWire(
            source_tag=options.get('source_tag', ''),
            batch_size=options.get('batch_size', limits.DEFAULT_BATCH_SIZE),
            concurrency=options.get('concurrency', limits.DEFAULT_CONCURRENCY),
        ),
        progress=BulkIngestProgress(
            **progress,
            images_per_s=state.get('images_per_s'),
            images_per_s_recent=state.get('images_per_s_recent'),
            eta_s=state.get('eta_s'),
            elapsed_s=state.get('elapsed_s'),
        ),
        stage_s=state.get('stage_s') or {},
        attempts=int(state.get('attempts') or 1),
        can_cancel=status in ACTIVE_STATUSES,
        can_resume=status in RESUMABLE_STATUSES,
        cancel_requested=status in ACTIVE_STATUSES and store.job.cancel_requested(),
        error=state.get('error'),
        started_at=state.get('started_at'),
        finished_at=state.get('finished_at'),
        poll_after_s=POLL_AFTER_S if status in ACTIVE_STATUSES else None,
    )


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
    """Start a run over a folder or manifest, or resume the interrupted, cancelled
    or partly failed run of the same source (``resumed: true``). One run per
    project at a time (409 ``bulk_ingest_busy``)."""
    source = BulkSource(
        'folder' if body.folder is not None else 'manifest',
        body.folder or body.manifest or '',
        normalise_extensions(body.extensions),
    )
    try:
        validate_source(source)
    except InvalidSourceError as exc:
        raise api_error(422, 'bulk_ingest_invalid_source', str(exc), project=_slug()) from None
    await _ensure_indexes(opensearch)
    # Refuse up front (503) what the first batch would refuse: no detector, no encoder.
    await _get_ingest_service(opensearch, registry)
    address = _own_address(request)
    options = runner.BulkOptions(body.source, body.batch_size, body.concurrency)
    try:
        store, resumed = runner.claim_run(source, options)
    except runner.BulkBusyError as exc:
        raise api_error(
            409, 'bulk_ingest_busy', str(exc), run_id=exc.run_id, project=_slug()
        ) from None
    batch_sender = sender.loopback_sender(
        address, api_prefix=get_curation_config().api_prefix, project=_slug()
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
    return run_wire(_open(run_id))


@router.get('/ingest/bulk/{run_id}/failures', response_model=BulkIngestFailurePage)
async def get_bulk_ingest_failures(
    run_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> BulkIngestFailurePage:
    """The failed images of the run's latest attempt, with the reason for each."""
    rows, total = _open(run_id).failures(offset=offset, limit=limit)
    return BulkIngestFailurePage(
        items=[BulkIngestFailure(**row) for row in rows], total=total, offset=offset, limit=limit
    )


@router.post('/ingest/bulk/{run_id}/cancel', response_model=BulkIngestRun)
async def cancel_bulk_ingest(run_id: str) -> BulkIngestRun:
    """Stop after the batches in flight; images already stored stay stored and a
    later start of the same source continues from them."""
    store = _open(run_id)
    runner.cancel_run(store)
    return run_wire(store)
