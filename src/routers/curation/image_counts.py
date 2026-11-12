"""Curation router sub-module: start the per-image count backfill.

``POST /curation/projects/{project}/image_counts/backfill`` answers 202 with the job id (409
``job_busy`` while one is active). The job stores the item counts of every image so the gallery
can sort by "most unlabelled first" and filter by label state without joining the items. New
images are counted by the periodic drain on their own; run this once on a project that already
had images. Progress and the final count are on ``GET .../jobs/{job_id}``.
"""

from __future__ import annotations

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep
from src.routers.curation._common import router
from src.schemas.curation.image_counts import ImageCountsBackfillStarted
from src.services.jobs.service import start_job


KIND = 'image_counts_backfill'
REQUESTER = 'curation-api'


@router.post(
    '/image_counts/backfill',
    response_model=ImageCountsBackfillStarted,
    status_code=202,
    tags=['Images'],
)
async def start_image_counts_backfill(session: DbSessionDep) -> ImageCountsBackfillStarted:
    """Count the items of every stored image (the gallery's served sort and label-state filter)."""
    started = await start_job(
        session, KIND, current_project_id(), params={}, requested_by=REQUESTER
    )
    return ImageCountsBackfillStarted(job_id=str(started.job.id), kind=KIND)
