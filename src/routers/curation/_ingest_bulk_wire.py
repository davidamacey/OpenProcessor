"""Shared pieces of the bulk-ingest routes: the bound project's slug and the run wire view
(used by ``/ingest/bulk*`` and the ``/ingest/sources*`` run controls)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.routers.curation._api_error import api_error
from src.routers.curation._ingest_bulk_models import (
    BulkIngestProgress,
    BulkIngestRun,
    BulkOptionsWire,
    BulkSourceWire,
)
from src.services.curation.bulk_ingest import limits
from src.services.curation.bulk_ingest.runs import get_run
from src.services.curation.bulk_ingest.source_runs import (
    ACTIVE_RUN_STATUSES,
    RESUMABLE_RUN_STATUSES,
)


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.jobs import JobView

POLL_AFTER_S = 2.0
_PROGRESS_KEYS = {
    'discovered': 0,
    'discovery_complete': False,
    'ingested': 0,
    'duplicates': 0,
    'failed': 0,
    'crops_indexed': 0,
    'n_embedded': 0,
    'n_not_embedded': 0,
    'n_embed_failed': 0,
    'in_flight': 0,
}
LIVE_STATUSES = ACTIVE_RUN_STATUSES | {'interrupted'}


def slug() -> str:
    return get_curation_config().project_slug


async def open_run(session: AsyncSession, project: uuid.UUID, run_id: str) -> JobView:
    run = await get_run(session, project, run_id)
    if run is None:
        raise api_error(404, 'bulk_ingest_not_found', f'no bulk ingest {run_id!r}', project=slug())
    return run


def _iso(value: Any) -> str | None:
    return None if value is None else value.isoformat()


def run_wire(run: JobView, *, resumed: bool = False) -> BulkIngestRun:
    params: dict[str, Any] = run.params or {}
    source = params.get('source') or {}
    options = params.get('options') or {}
    counters = {**_PROGRESS_KEYS, **(run.progress.get('counters') or {})}
    progress = {key: counters[key] for key in _PROGRESS_KEYS}
    return BulkIngestRun(
        project=slug(),
        run_id=run.id,
        status=run.status,
        resumed=resumed,
        source=BulkSourceWire(
            kind=source.get('kind', 'folder'),
            path=source.get('path', ''),
            extensions=list(source.get('extensions') or []),
            label=source.get('label'),
        ),
        options=BulkOptionsWire(
            source_tag=options.get('source_tag', ''),
            batch_size=options.get('batch_size', limits.DEFAULT_BATCH_SIZE),
            concurrency=options.get('concurrency', limits.DEFAULT_CONCURRENCY),
        ),
        progress=BulkIngestProgress(
            **progress,
            skipped=0,
            processed=int(run.progress.get('done') or 0),
            images_per_s=counters.get('images_per_s'),
            images_per_s_recent=counters.get('images_per_s_recent'),
            eta_s=counters.get('eta_s'),
            elapsed_s=counters.get('elapsed_s'),
        ),
        stage_s=counters.get('stage_s') or {},
        attempts=max(run.attempt, 1),
        can_cancel=run.status in {'queued', 'running', 'interrupted'},
        can_resume=run.status in RESUMABLE_RUN_STATUSES,
        cancel_requested=run.cancel_requested and run.status in LIVE_STATUSES,
        error=run.error_message,
        started_at=_iso(run.started_at),
        finished_at=_iso(run.finished_at),
        poll_after_s=POLL_AFTER_S if run.status in LIVE_STATUSES else None,
    )
