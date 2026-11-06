"""The newest job of a kind in the bound project: what the "current or last run" status routes serve."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project_id
from src.services.jobs import status


if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.jobs import JobView

ACTIVE_STATUSES = frozenset({'queued', 'running', 'paused', 'cancelling'})


async def latest_job(session: AsyncSession, kind: str) -> JobView | None:
    """The kind's newest job with its parameters, or ``None`` when the project never ran one."""
    project = current_project_id()
    page = await status.list_jobs(session, project=project, kinds=[kind], limit=1)
    if not page.items:
        return None
    return await status.get_job(session, uuid.UUID(page.items[0].id), project=project)


def epoch(stamp: datetime | None) -> float:
    """Seconds since the epoch for the status wires that serve floats; 0.0 before the job started."""
    return 0.0 if stamp is None else stamp.timestamp()


def common_state(view: JobView | None) -> dict[str, Any]:
    """``job_id`` / ``status`` / the two times / ``error`` of a job row, ``idle`` without one."""
    if view is None:
        return {
            'job_id': '',
            'status': 'idle',
            'started_at': 0.0,
            'finished_at': 0.0,
            'error': None,
        }
    return {
        'job_id': view.id,
        'status': view.status,
        'started_at': epoch(view.started_at),
        'finished_at': epoch(view.finished_at),
        'error': view.error_message,
    }


async def cancel_latest(session: AsyncSession, kind: str) -> bool:
    """Cancel the kind's newest job if it is still active; whether there was one to cancel."""
    from src.services.jobs.service import cancel_job

    current = await latest_job(session, kind)
    if current is None or current.status not in ACTIVE_STATUSES:
        return False
    await cancel_job(session, uuid.UUID(current.id), current_project_id(), requested_by='api')
    return True
