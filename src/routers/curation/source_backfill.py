"""Curation router sub-module: start the source backfill.

``POST /curation/projects/{project}/source_backfill/runs`` answers 202 with the job id (409
``job_busy`` while one is active). The job records the exact-content key of every stored image
that has none, which the 0.6.0 upgrade report asks for; progress and the final counts are on
``GET .../jobs/{job_id}``.
"""

from __future__ import annotations

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._common import router
from src.schemas.curation.source_backfill import SourceBackfillStarted
from src.services.jobs.service import start_job


KIND = 'source_backfill'
REQUESTER = 'curation-api'


@router.post(
    '/source_backfill/runs',
    response_model=SourceBackfillStarted,
    status_code=202,
    tags=['Source backfill'],
)
async def start_source_backfill(session: DbSessionDep) -> SourceBackfillStarted:
    """Start the pass that reads each stored image once and records its content key."""
    started = await start_job(
        session, KIND, current_project_id(), params={}, requested_by=REQUESTER
    )
    return SourceBackfillStarted(job_id=str(started.job.id), kind=KIND)
