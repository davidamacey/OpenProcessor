"""Start, read, cancel and pause the auto-label job of the bound project.

The job is a ``jobs`` row of kind ``autolabel`` (exclusive per project); this module is the
family's view of it. ``state_of`` maps the row to the state the ``/pipeline/auto_label*`` routes
have always served, so a client reads the same fields.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project_id
from src.services.jobs import service as jobs, status as job_status
from src.services.jobs.service import rejection


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.jobs import JobView

KIND = 'autolabel'
PIPELINE = 'src.services.curation.auto_label_run:run_auto_label'
QUEUE_MAX_AGE_S = 3600.0
IDLE: dict[str, Any] = {
    'job_id': '',
    'status': 'idle',
    'stage': '',
    'processed': 0,
    'total': 0,
    'started_at': 0.0,
    'finished_at': 0.0,
    'error': None,
    'error_detail': None,
    'result': {},
    'args': {},
    'pipeline': '',
    'backend': None,
    'backend_detail': None,
    'free_vram_mb': None,
    'peak_vram_mb': None,
    'stage_durations': {},
    'eta_seconds': None,
    'elapsed_seconds': 0.0,
    'queue_age_seconds': None,
    'queue_max_age_seconds': QUEUE_MAX_AGE_S,
}


def _epoch(value: Any) -> float:
    return 0.0 if value is None else float(value.timestamp())


def state_of(view: JobView) -> dict[str, Any]:
    """The route state of one job row."""
    progress = view.progress or {}
    counters = progress.get('counters') or {}
    processed, total = int(progress.get('done') or 0), int(progress.get('total') or 0)
    started, finished = _epoch(view.started_at), _epoch(view.finished_at)
    running = view.status == 'running'
    now = _epoch(view.updated_at)
    elapsed = max(0.0, now - started) if running and started else 0.0
    eta = None
    if running and processed > 0 and total > processed:
        eta = round(elapsed / processed * (total - processed), 1)
    queued = view.status == 'queued'
    return {
        **IDLE,
        'job_id': view.id,
        'status': view.status,
        'stage': progress.get('stage') or '',
        'processed': processed,
        'total': total,
        'started_at': started,
        'finished_at': finished,
        'error': view.error_message,
        'result': dict(view.result or {}),
        'args': dict(view.params or {}),
        'pipeline': PIPELINE,
        'backend': counters.get('backend'),
        'backend_detail': counters.get('backend_detail'),
        'free_vram_mb': counters.get('free_vram_mb'),
        'peak_vram_mb': counters.get('peak_vram_mb'),
        'stage_durations': dict(counters.get('stage_durations') or {}),
        'eta_seconds': eta,
        'elapsed_seconds': round(elapsed, 1),
        'queue_age_seconds': (
            round(max(0.0, now - _epoch(view.queued_at or view.created_at)), 1) if queued else None
        ),
        'pause_reason': view.pause_reason,
    }


async def start(
    session: AsyncSession, params: Mapping[str, Any], *, idempotency_key: str | None = None
) -> dict[str, Any]:
    """Queue a run; 409 ``job_busy`` while one is queued, running or paused."""
    result = await jobs.start_job(
        session, KIND, current_project_id(), params=params, idempotency_key=idempotency_key
    )
    view = await job_status.get_job(session, result.job.id, project=current_project_id())
    return state_of(view)


async def current(session: AsyncSession) -> dict[str, Any]:
    """The newest run's state, or the idle state when the project never ran one."""
    page = await job_status.list_jobs(session, project=current_project_id(), kinds=(KIND,), limit=1)
    if not page.items:
        return dict(IDLE)
    return await by_id(session, page.items[0].id)


async def by_id(session: AsyncSession, job_id: str) -> dict[str, Any]:
    try:
        parsed = uuid.UUID(job_id)
    except ValueError:
        raise rejection(404, 'job_not_found', f'unknown auto_label job {job_id!r}') from None
    view = await job_status.get_job(session, parsed, project=current_project_id())
    if view.kind != KIND:
        raise rejection(404, 'job_not_found', f'unknown auto_label job {job_id!r}')
    return state_of(view)


async def _active_id(session: AsyncSession) -> uuid.UUID | None:
    page = await job_status.list_jobs(
        session,
        project=current_project_id(),
        kinds=(KIND,),
        statuses=('queued', 'running', 'paused', 'cancelling'),
        limit=1,
    )
    return uuid.UUID(page.items[0].id) if page.items else None


async def cancel(session: AsyncSession, *, requested_by: str) -> bool:
    """Cancel the active run; ``False`` when there is none."""
    job_id = await _active_id(session)
    if job_id is None:
        return False
    await jobs.cancel_job(session, job_id, current_project_id(), requested_by=requested_by)
    return True


async def pause(session: AsyncSession) -> bool:
    job_id = await _active_id(session)
    if job_id is None:
        return False
    await jobs.pause_job(session, job_id, current_project_id())
    return True


async def resume(session: AsyncSession) -> bool:
    page = await job_status.list_jobs(
        session,
        project=current_project_id(),
        kinds=(KIND,),
        statuses=('paused', 'interrupted'),
        limit=1,
    )
    if not page.items:
        return False
    await jobs.resume_job(session, uuid.UUID(page.items[0].id), current_project_id())
    return True
