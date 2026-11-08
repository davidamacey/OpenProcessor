"""Curation auto-label job control endpoints (status, cancel, pause, resume).

Side-effect import: registers ``@router`` handlers on the shared ``_common.router``. The job is a
``jobs`` row; these routes are the family's view of it.
"""

from __future__ import annotations

from typing import Any

from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._autolabel_models import (
    AutoLabelActionResponse,
    AutoLabelCancelResponse,
    AutoLabelJobState,
)
from src.routers.curation._common import router
from src.services.curation.autolabel import service


@router.get(
    '/pipeline/auto_label/status',
    response_model=None,
    responses={200: {'model': AutoLabelJobState}},
)
async def pipeline_auto_label_status(session: DbSessionDep) -> dict[str, Any]:
    """Poll the current/last auto_label job state."""
    return await service.current(session)


@router.get(
    '/pipeline/auto_label/status/{job_id}',
    response_model=None,
    responses={200: {'model': AutoLabelJobState}},
)
async def pipeline_auto_label_job_status(session: DbSessionDep, job_id: str) -> dict[str, Any]:
    """State of one auto_label job by the ``job_id`` its start returned (404 for another id)."""
    return await service.by_id(session, job_id)


@router.post(
    '/pipeline/auto_label/cancel',
    response_model=None,
    responses={200: {'model': AutoLabelCancelResponse}},
)
async def pipeline_auto_label_cancel(session: DbSessionDep) -> dict[str, Any]:
    """Cancel the active auto_label job: a queued one at once, a running one at its next boundary."""
    cancelled = await service.cancel(session, requested_by='api')
    return {'cancelled': cancelled, **await service.current(session)}


@router.post(
    '/pipeline/auto_label/pause',
    response_model=None,
    responses={200: {'model': AutoLabelActionResponse}},
)
async def pipeline_auto_label_pause(session: DbSessionDep) -> dict[str, Any]:
    """Pause the active job between stages or VLM chunks; chunks in flight finish."""
    done = await service.pause(session)
    return {'done': done, **await service.current(session)}


@router.post(
    '/pipeline/auto_label/resume',
    response_model=None,
    responses={200: {'model': AutoLabelActionResponse}},
)
async def pipeline_auto_label_resume(session: DbSessionDep) -> dict[str, Any]:
    """Resume the paused job (409 ``resume_blocked`` names why it cannot yet)."""
    done = await service.resume(session)
    return {'done': done, **await service.current(session)}
