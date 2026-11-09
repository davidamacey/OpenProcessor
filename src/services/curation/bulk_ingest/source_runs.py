"""Start runs of a registered source and describe where it stands.

``start_source_run`` is the one way a source's queue becomes a bulk run: the routes (run now,
one-shot starts) and the dispatcher (poll, schedule, webhook) all call it, so there is one
runner, one claim and one set of per-project rules: ``bulk_ingest_run`` is exclusive per project.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.core.exceptions import RequestRejectedError
from src.db.repos import ingest_sources as repo
from src.db.session import project_tx
from src.services.curation.bulk_ingest.s3_listing import LAST_SCAN_META, SCAN_META
from src.services.curation.bulk_ingest.schedule import Cron
from src.services.curation.bulk_ingest.source_queue import path_of, version_of
from src.services.curation.bulk_ingest.sources import IngestSource, source_queue
from src.services.jobs.service import start_job
from src.services.jobs.status import get_job


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.jobs import JobView

ACTIVE_RUN_STATUSES = frozenset({'queued', 'running', 'paused', 'cancelling'})
# A finished run that stopped with work left is continued by starting its source again.
RESUMABLE_RUN_STATUSES = frozenset({'interrupted', 'failed', 'cancelled', 'completed_with_errors'})
KIND = 'bulk_ingest_run'
SOURCE_POLL_KIND = 'source_poll'


class BulkBusyError(Exception):
    def __init__(self, run_id: str | None) -> None:
        super().__init__(f'a bulk ingest is already running: {run_id}')
        self.run_id = run_id


def run_params(source: IngestSource, *, scan: bool, fresh: bool) -> dict[str, Any]:
    """What the job needs to run, and a snapshot of the source for the run's wire view."""
    direct = source.one_shot and source.backend != 's3'  # the wire names a bucket start a queue
    return {
        'source_id': source.source_id,
        'scan': scan,
        'fresh': fresh,
        'options': {
            'source_tag': source.source_tag,
            'batch_size': source.batch_size,
            'concurrency': source.concurrency,
        },
        'source': {
            'kind': source.backend if direct else 'queue',
            'path': source.location if direct else source.source_id,
            'extensions': list(source.extensions),
            'label': None if source.one_shot else f'{source.name} ({source.location})',
        },
    }


async def start_source_run(
    session: AsyncSession,
    project: uuid.UUID,
    source: IngestSource,
    *,
    scan: bool,
    fresh: bool = False,
) -> uuid.UUID:
    """Queue a run draining ``source``'s queue (after a scan when ``scan``); returns its job id.

    Raises :class:`BulkBusyError` while any run of the project is queued, running or paused.
    ``fresh`` starts a one-shot source over: its finished files are queued again."""
    try:
        started = await start_job(
            session, KIND, project, params=run_params(source, scan=scan, fresh=fresh)
        )
    except RequestRejectedError as exc:
        if exc.detail.get('error') != 'job_busy':
            raise
        raise BulkBusyError(exc.detail.get('active_job_id')) from None
    async with project_tx(session, project):
        await session.run_sync(
            lambda sync: repo.state_update(
                sync, uuid.UUID(source.source_id), {'last_run_id': str(started.job.id)}
            )
        )
    return started.job.id


async def start_poll(session: AsyncSession, project: uuid.UUID, source: IngestSource) -> bool:
    """Queue the poll job of ``source``; ``False`` when its last poll is still queued or running."""
    try:
        await start_job(session, SOURCE_POLL_KIND, project, params={'source_id': source.source_id})
    except RequestRejectedError as exc:
        if exc.detail.get('error') != 'job_busy':
            raise
        return False
    return True


async def last_run(
    session: AsyncSession, project: uuid.UUID, state: dict[str, Any]
) -> JobView | None:
    """The job of the source's last run, if it has one and it still exists."""
    run_id = state.get('last_run_id')
    parsed = None
    if isinstance(run_id, str):
        try:
            parsed = uuid.UUID(run_id)
        except ValueError:
            parsed = None
    if parsed is None:
        return None
    try:
        return await get_job(session, parsed, project=project)
    except RequestRejectedError:
        return None  # pruned


def _iso(epoch: float | None) -> str | None:
    return datetime.fromtimestamp(epoch, UTC).isoformat() if epoch else None


def next_poll_at(source: IngestSource, state: dict[str, Any]) -> float | None:
    if not source.watch.enabled:
        return None
    last = state.get('last_poll_started_at')
    due = (last + source.watch.interval_s) if last else time.time()
    return max(due, state.get('backoff_until') or 0.0)


def next_fire_at(source: IngestSource, state: dict[str, Any]) -> str | None:
    if not source.schedule.enabled:
        return None
    last = state.get('last_fire_at')
    after = datetime.fromisoformat(last) if last else datetime.now(UTC)
    return Cron.parse(source.schedule.cron).next_after(after).isoformat()


def run_summary(job: JobView | None) -> dict[str, Any] | None:
    if job is None:
        return None
    counters = job.progress.get('counters') or {}
    return {
        'run_id': job.id,
        'status': job.status,
        'started_at': None if job.started_at is None else job.started_at.isoformat(),
        'finished_at': None if job.finished_at is None else job.finished_at.isoformat(),
        'ingested': counters.get('ingested', 0),
        'duplicates': counters.get('duplicates', 0),
        'failed': counters.get('failed', 0),
        'error': job.error_message,
    }


def run_is_live(summary: dict[str, Any] | None) -> bool:
    return summary is not None and summary['status'] in ACTIVE_RUN_STATUSES


async def source_status(
    session: AsyncSession, project: uuid.UUID, source: IngestSource
) -> dict[str, Any]:
    """Everything the UI shows for a source, computed here so the frontend only renders."""
    queue = source_queue(source)

    def read() -> tuple[dict[str, int], dict[str, Any], list[dict[str, Any]]]:
        return queue.counts(), queue.meta_all(), queue.errors(limit=20)

    found, state, errors = await asyncio.to_thread(read)
    # a claimed file is still to be ingested: the view has no state for it
    counts = {
        'pending': found['pending'] + found['claimed'],
        'done': found['done'],
        'failed': found['failed'],
        'dead': found['dead'],
    }
    scan_progress = state.get(SCAN_META)
    return {
        'counts': counts,
        'scan': {
            'active': scan_progress is not None,
            'pages': (scan_progress or {}).get('pages', 0),
            'listed': (scan_progress or {}).get('listed', 0),
            'queued': (scan_progress or {}).get('queued', 0),
            'last': state.get(LAST_SCAN_META),
        },
        'watch': {
            'last_poll_at': _iso(state.get('last_poll_started_at')),
            'next_poll_at': _iso(next_poll_at(source, state)),
            'error': state.get('last_poll_error'),
            'backoff_until': _iso(state.get('backoff_until')),
        },
        'schedule': {
            'last_fire_at': state.get('last_fire_at'),
            'next_fire_at': next_fire_at(source, state),
        },
        'webhook': {
            'received': state.get('events.received', 0),
            'queued': state.get('events.queued', 0),
            'duplicates': state.get('events.duplicates', 0),
            'ignored': state.get('events.ignored', 0),
            'last_event_at': _iso(state.get('events.last_at')),
        },
        'last_run': run_summary(await last_run(session, project, state)),
        'errors': [{**e, 'key': path_of(e['key']), 'etag': version_of(e['key'])} for e in errors],
    }
