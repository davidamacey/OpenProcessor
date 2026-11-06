"""Curation router sub-module: the project's jobs, read only.

``GET /curation/projects/{project}/jobs`` lists them (filter by kind and status, keyset pages) and
``.../jobs/{job_id}`` shows one. Every job of every family has a row, so this is the one place to
see what a project is running; the per-family status routes keep their own views of the same row.
Starting, cancelling and pausing stay on the family routes, which call the job service.
"""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves the path parameter type
from typing import Annotated

from fastapi import Query

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._common import router
from src.schemas.jobs import JobPage, JobView
from src.services.jobs import status


@router.get('/jobs', response_model=JobPage)
async def list_project_jobs(
    session: DbSessionDep,
    kind: Annotated[list[str] | None, Query(description='Only these kinds; repeat to add.')] = None,
    status_filter: Annotated[
        list[str] | None, Query(alias='status', description='Only these statuses; repeat to add.')
    ] = None,
    limit: Annotated[int, Query(ge=1, le=status.MAX_PAGE)] = status.DEFAULT_PAGE,
    cursor: Annotated[str | None, Query(description='`next_cursor` of the previous page.')] = None,
) -> JobPage:
    """The project's jobs, newest first."""
    return await status.list_jobs(
        session,
        project=current_project_id(),
        kinds=kind or (),
        statuses=status_filter or (),
        limit=limit,
        cursor=cursor,
    )


@router.get('/jobs/{job_id}', response_model=JobView)
async def get_project_job(session: DbSessionDep, job_id: uuid.UUID) -> JobView:
    """One job of the project, with its parameters. 404 `job_not_found` for another project's id."""
    return await status.get_job(session, job_id, project=current_project_id())
