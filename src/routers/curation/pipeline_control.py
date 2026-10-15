"""Curation auto-label job control endpoints (status and cancel).

Split out of :mod:`pipeline` to keep that module under the 700-LOC hook
ceiling. Side-effect import: registers ``@router`` handlers on the shared
``_common.router``.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.routers.curation._autolabel_models import AutoLabelCancelResponse, AutoLabelJobState
from src.routers.curation._common import router


@router.get(
    '/pipeline/auto_label/status',
    response_model=None,
    responses={200: {'model': AutoLabelJobState}},
)
async def pipeline_auto_label_status() -> dict[str, Any]:
    """Poll the current/last auto_label job state."""
    from src.services.curation.autolabel import job as auto_label_job

    return auto_label_job.get_state()


@router.get(
    '/pipeline/auto_label/status/{job_id}',
    response_model=None,
    responses={200: {'model': AutoLabelJobState}},
)
async def pipeline_auto_label_job_status(job_id: str) -> dict[str, Any]:
    """State of one auto_label job by the ``job_id`` its start returned.

    Same shape as ``GET /pipeline/auto_label/status``. Answers for the job
    even after a later job replaced it as the current one (its final
    state is kept; the newest 50 are retained). ``404`` for an unknown id.
    """
    from src.services.curation.autolabel import job as auto_label_job

    state = auto_label_job.get_job_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f'unknown auto_label job {job_id!r}')
    return state


@router.post(
    '/pipeline/auto_label/cancel',
    response_model=None,
    responses={200: {'model': AutoLabelCancelResponse}},
)
async def pipeline_auto_label_cancel() -> dict[str, Any]:
    """Request cancellation of the active auto_label job."""
    from src.services.curation.autolabel import job as auto_label_job

    cancelled = auto_label_job.cancel_job()
    return {'cancelled': cancelled, **auto_label_job.get_state()}
