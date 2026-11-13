"""Job reads for the job routes: list, one job, and the row-to-wire serializer.

Per-family status routes stay the family views and read the same row. A project's routes run in
the project's scope (a project sees only its own jobs); the deployment-level list runs in
deployment scope and filters to the rows that belong to no project.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import uuid
from typing import TYPE_CHECKING

from src.db.repos import jobs_listing as listing
from src.db.session import deployment_tx, project_tx
from src.schemas.jobs import JobPage, JobView
from src.services.jobs.service import rejection


if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

DEFAULT_PAGE = 50
MAX_PAGE = 200


def to_view(
    listed: listing.ListedJob, *, with_params: bool = False, queue_ahead: int | None = None
) -> JobView:
    row = listed.row
    return JobView(
        id=str(row.id),
        kind=row.kind,
        project=listed.project_slug,
        status=row.status,
        scope=row.scope,
        queue=row.queue,
        attempt=row.attempt,
        retry_count=row.retry_count,
        max_attempts=row.max_attempts,
        pause_reason=row.pause_reason,
        pause_requested=row.pause_requested_at is not None,
        cancel_requested=row.cancel_requested_at is not None,
        progress=dict(row.progress or {}),
        progress_updated_at=row.progress_updated_at,
        result=None if row.result is None else dict(row.result),
        error_code=row.error_code,
        error_message=row.error_message,
        params=dict(row.params or {}) if with_params else None,
        queue_ahead=queue_ahead,
        requested_by=row.requested_by,
        created_at=row.created_at,
        queued_at=row.queued_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        updated_at=row.updated_at,
    )


def encode_cursor(created_at: dt.datetime, job_id: uuid.UUID) -> str:
    raw = json.dumps([created_at.isoformat(), str(job_id)]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip('=')


def decode_cursor(cursor: str) -> tuple[dt.datetime, uuid.UUID]:
    try:
        padded = cursor + '=' * (-len(cursor) % 4)
        stamp, ident = json.loads(base64.urlsafe_b64decode(padded))
        return dt.datetime.fromisoformat(stamp), uuid.UUID(ident)
    except (ValueError, TypeError):
        raise rejection(
            422, 'invalid_cursor', 'the cursor is not one this route returned'
        ) from None


def _transaction(
    session: AsyncSession, project: uuid.UUID | None
) -> AbstractAsyncContextManager[AsyncSession]:
    if project is None:
        return deployment_tx(session, 'list deployment-level jobs')
    return project_tx(session, project)


async def list_jobs(
    session: AsyncSession,
    *,
    project: uuid.UUID | None,
    kinds: Sequence[str] = (),
    statuses: Sequence[str] = (),
    limit: int = DEFAULT_PAGE,
    cursor: str | None = None,
) -> JobPage:
    """A project's jobs (``project`` set) or the deployment-level jobs (``None``), newest first."""
    after = None if cursor is None else decode_cursor(cursor)
    flt = listing.JobFilter(project, tuple(kinds), tuple(statuses))
    size = min(max(limit, 1), MAX_PAGE)
    async with _transaction(session, project):
        found = await session.run_sync(
            lambda sync: listing.list_page(sync, flt, limit=size + 1, after=after)
        )
    page, more = found[:size], len(found) > size
    next_cursor = None
    if more and page:
        last = page[-1].row
        next_cursor = encode_cursor(last.created_at, last.id)
    return JobPage(items=[to_view(j) for j in page], next_cursor=next_cursor)


async def get_job(
    session: AsyncSession, job_id: uuid.UUID, *, project: uuid.UUID | None
) -> JobView:
    flt = listing.JobFilter(project)
    async with _transaction(session, project):
        found = await session.run_sync(lambda sync: listing.get_one(sync, flt, job_id))
    if found is None:
        raise rejection(404, 'job_not_found', f'no job {job_id}')
    row = found.row
    queued_at = row.queued_at
    ahead = None
    if row.status == 'queued' and queued_at is not None:
        async with deployment_tx(session, 'count the jobs queued ahead of one on its queue'):
            ahead = await session.run_sync(
                lambda sync: listing.queued_ahead(
                    sync, queue=row.queue, queued_at=queued_at, job_id=row.id
                )
            )
    return to_view(found, with_params=True, queue_ahead=ahead)
