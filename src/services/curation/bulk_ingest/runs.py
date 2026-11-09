"""Bulk-ingest runs as the jobs table keeps them: one ``bulk_ingest_run`` row per run."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from src.core.exceptions import RequestRejectedError
from src.db.repos import ingest_sources as repo
from src.db.session import project_tx
from src.services.curation.bulk_ingest.source_runs import ACTIVE_RUN_STATUSES, KIND
from src.services.jobs.status import MAX_PAGE, get_job, list_jobs


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.jobs import JobView


def parse_run_id(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


async def get_run(session: AsyncSession, project: uuid.UUID, run_id: str) -> JobView | None:
    """The run ``run_id`` of ``project``, or ``None`` when no such run exists in it."""
    parsed = parse_run_id(run_id)
    if parsed is None:
        return None
    try:
        job = await get_job(session, parsed, project=project)
    except RequestRejectedError:
        return None
    return job if job.kind == KIND else None


async def list_runs(session: AsyncSession, project: uuid.UUID) -> list[JobView]:
    """Every run of the project, newest first."""
    runs: list[JobView] = []
    cursor: str | None = None
    while True:
        page = await list_jobs(
            session, project=project, kinds=[KIND], limit=MAX_PAGE, cursor=cursor
        )
        runs.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            return runs


async def live_run(session: AsyncSession, project: uuid.UUID) -> JobView | None:
    """The run of the project that is queued, running or being cancelled, if any."""
    page = await list_jobs(
        session,
        project=project,
        kinds=[KIND],
        statuses=sorted(ACTIVE_RUN_STATUSES),
        limit=1,
    )
    return page.items[0] if page.items else None


async def failures(
    session: AsyncSession, project: uuid.UUID, run: JobView, *, offset: int, limit: int
) -> tuple[list[dict[str, str]], int]:
    """``(page, total)`` of the files the run left failed."""
    source = uuid.UUID(str(run.params['source_id'])) if run.params else None
    if source is None:
        return [], 0
    job = uuid.UUID(run.id)
    async with project_tx(session, project):
        return await session.run_sync(
            lambda sync: repo.failures(sync, source, job, offset=offset, limit=limit)
        )
