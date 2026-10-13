"""Shared pieces of the bulk-ingest routes: the bound project's slug, the API's own
listening address, and the run wire view (used by ``/ingest/bulk*`` and the
``/ingest/sources*`` run controls)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from src.config import get_curation_config
from src.routers.curation._config_common_models import api_error
from src.routers.curation._ingest_bulk_models import (
    BulkIngestProgress,
    BulkIngestRun,
    BulkOptionsWire,
    BulkSourceWire,
)
from src.services.curation.bulk_ingest import limits
from src.services.curation.bulk_ingest.store import (
    ACTIVE_STATUSES,
    RESUMABLE_STATUSES,
    BulkStore,
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


def slug() -> str:
    return get_curation_config().project_slug


def own_address(request: Request) -> tuple[str, int]:
    """The API's own listening socket, which batches are sent back to so they spread
    over every worker process."""
    server = request.scope.get('server')
    if not server or server[1] is None:
        raise HTTPException(
            status_code=503, detail='bulk ingest needs the API to listen on a TCP socket'
        )
    return str(server[0]), int(server[1])


def open_run(run_id: str) -> BulkStore:
    store = open_store(run_id)
    if store is None:
        raise api_error(404, 'bulk_ingest_not_found', f'no bulk ingest {run_id!r}', project=slug())
    return store


def run_wire(store: BulkStore, *, resumed: bool = False) -> BulkIngestRun:
    state: dict[str, Any] = store.repaired_state()
    status = state.get('status', 'queued')
    source = state.get('source') or {}
    options = state.get('options') or {}
    progress = {key: state.get(key, default) for key, default in _PROGRESS_KEYS.items()}
    return BulkIngestRun(
        project=slug(),
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
