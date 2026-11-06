"""``GET /jobs``: the deployment-level jobs, the ones that belong to no project (read only)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.schemas.jobs import JobPage
from src.services.jobs import status


router = APIRouter(prefix='/jobs', tags=['jobs'])


@router.get('', response_model=JobPage)
async def list_deployment_jobs(
    session: DbSessionDep,
    kind: Annotated[list[str] | None, Query(description='Only these kinds; repeat to add.')] = None,
    status_filter: Annotated[
        list[str] | None, Query(alias='status', description='Only these statuses; repeat to add.')
    ] = None,
    limit: Annotated[int, Query(ge=1, le=status.MAX_PAGE)] = status.DEFAULT_PAGE,
    cursor: Annotated[str | None, Query(description='`next_cursor` of the previous page.')] = None,
) -> JobPage:
    """Deployment-wide jobs (project delete, directory ingest...), newest first."""
    return await status.list_jobs(
        session,
        project=None,
        kinds=kind or (),
        statuses=status_filter or (),
        limit=limit,
        cursor=cursor,
    )
