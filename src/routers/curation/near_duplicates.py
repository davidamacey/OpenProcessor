"""Curation router sub-module: near-duplicate status, runs, groups and decisions.

Thin adapters over ``services/near_duplicates/status.py``. A run is a job: ``POST .../runs``
answers 202 with the job id (409 ``job_busy`` while one is active for that level). Group ids on
the wire are lineage ids. The profile routes are the ``near_dup_profiles`` family of
``research_configs.py``.
"""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves the path parameter type
from typing import Annotated

from fastapi import Query, Response

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._common import router
from src.schemas.curation import near_duplicates as wire
from src.services.near_duplicates import status as service


TAG = 'Near-duplicates'
REQUESTER = 'curation-api'


@router.get('/near_duplicates/status', response_model=wire.NearDupStatus, tags=[TAG])
async def near_duplicate_status(session: DbSessionDep) -> wire.NearDupStatus:
    """Per level: the current run, the pending count and why it is stale, if it is."""
    return await service.status(session, current_project_id())


@router.post(
    '/near_duplicates/runs', response_model=wire.NearDupRunStarted, status_code=202, tags=[TAG]
)
async def start_near_duplicate_run(
    session: DbSessionDep, body: wire.NearDupRunRequest
) -> wire.NearDupRunStarted:
    """Start a full build (`mode` full) or the incremental pass (`incremental`) of one level."""
    return await service.start_run(session, current_project_id(), body, REQUESTER)


@router.get('/near_duplicates/runs', response_model=wire.NearDupRunPage, tags=[TAG])
async def list_near_duplicate_runs(
    session: DbSessionDep,
    level: Annotated[wire.Level | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=service.MAX_PAGE)] = service.DEFAULT_PAGE,
) -> wire.NearDupRunPage:
    return await service.runs(session, current_project_id(), level, limit)


@router.get('/near_duplicates/runs/{run_id}', response_model=wire.NearDupRunView, tags=[TAG])
async def get_near_duplicate_run(session: DbSessionDep, run_id: uuid.UUID) -> wire.NearDupRunView:
    return await service.run(session, current_project_id(), run_id)


@router.get('/near_duplicates/groups', response_model=wire.NearDupGroupPage, tags=[TAG])
async def list_near_duplicate_groups(
    session: DbSessionDep,
    level: Annotated[wire.Level, Query()],
    purpose: Annotated[wire.Purpose, Query()] = 'dedup',
    min_size: Annotated[int, Query(ge=2)] = 2,
    has_holdout: Annotated[bool | None, Query()] = None,
    cursor: Annotated[str | None, Query(description='`next_cursor` of the previous page.')] = None,
    limit: Annotated[int, Query(ge=1, le=service.MAX_PAGE)] = service.DEFAULT_PAGE,
) -> wire.NearDupGroupPage:
    """The current run's groups of one level, largest first."""
    return await service.groups(
        session,
        current_project_id(),
        level=level,
        purpose=purpose,
        min_size=min_size,
        has_holdout=has_holdout,
        cursor=cursor,
        limit=limit,
    )


@router.get(
    '/near_duplicates/groups/{group_id}', response_model=wire.NearDupGroupDetail, tags=[TAG]
)
async def get_near_duplicate_group(
    session: DbSessionDep, group_id: uuid.UUID, level: Annotated[wire.Level, Query()]
) -> wire.NearDupGroupDetail:
    return await service.group(session, current_project_id(), level, group_id)


@router.post(
    '/near_duplicates/decisions',
    response_model=wire.NearDupDecisionView,
    status_code=201,
    tags=[TAG],
)
async def decide_near_duplicates(
    session: DbSessionDep, body: wire.NearDupDecisionIn
) -> wire.NearDupDecisionView:
    """Pin two members together (`same`) or apart (`different`) for every later run."""
    return await service.decide(session, current_project_id(), body, REQUESTER)


@router.delete('/near_duplicates/decisions/{decision_id}', status_code=204, tags=[TAG])
async def undecide_near_duplicates(session: DbSessionDep, decision_id: uuid.UUID) -> Response:
    await service.undecide(session, current_project_id(), decision_id)
    return Response(status_code=204)
