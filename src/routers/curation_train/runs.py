"""Run status, listing, log tail and cancel routes under ``/train``."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path as PathParam, Query
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from src.services.training import jobs as train_jobs, promote_job
from src.services.training.job_models import TrainJobStatus


router = APIRouter(default_response_class=ORJSONResponse)


@router.get('/status', response_model=TrainJobStatus | None)
async def latest_status() -> TrainJobStatus | None:
    """Return the active or most-recent run's ``status.json``.

    Returns ``null`` (HTTP 200) if there are no runs yet.
    """
    try:
        return await train_jobs.get_active_job()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get('/status/{job_id}', response_model=TrainJobStatus)
async def status_by_id(
    job_id: Annotated[str, PathParam(description='Training job_id')],
) -> TrainJobStatus:
    """Return the ``status.json`` for a specific run.

    404 if no spec or status exists for that id.
    """
    try:
        result = await train_jobs.read_status(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f'job_id not found: {job_id}')
    return result.model_copy(update={'promote': promote_job.latest_for_run(job_id)})


class RunsListResponse(BaseModel):
    items: list[TrainJobStatus]
    total: int


@router.get('/runs', response_model=RunsListResponse)
async def list_runs_endpoint(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunsListResponse:
    """Paginated past-runs list (newest first)."""
    items = await train_jobs.list_runs(limit=limit, offset=offset)
    # ``list_runs`` is page-bounded; for total we re-query a larger chunk.
    # The labeler's infinite-scroll only needs ``len(items)`` to know if
    # there's more — we report that via the ``total`` field.
    deeper = await train_jobs.list_runs(limit=limit + offset + 1, offset=0)
    return RunsListResponse(items=items, total=len(deeper))


class LogTailResponse(BaseModel):
    job_id: str
    lines: list[str]


@router.get('/log/tail/{job_id}', response_model=LogTailResponse)
async def tail_log(
    job_id: Annotated[str, PathParam(description='Training job_id')],
    lines: Annotated[int, Query(ge=1, le=5000)] = 200,
) -> LogTailResponse:
    """Return the last N lines of the run log."""
    try:
        out = await train_jobs.tail_run_log(job_id, lines=lines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return LogTailResponse(job_id=job_id, lines=out)


class CancelResponse(BaseModel):
    cancelled: bool
    job_id: str


@router.post('/cancel/{job_id}', response_model=CancelResponse)
async def cancel_run(
    job_id: Annotated[str, PathParam(description='Training job_id')],
) -> CancelResponse:
    """Drop the cancel sentinel; trainer picks it up between epochs."""
    try:
        await train_jobs.write_cancel(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CancelResponse(cancelled=True, job_id=job_id)


class CancelCampaignResponse(BaseModel):
    cancelled: int
    campaign_id: str


@router.post('/cancel_campaign/{campaign_id}', response_model=CancelCampaignResponse)
async def cancel_campaign_endpoint(
    campaign_id: Annotated[str, PathParam(description='Campaign id')],
) -> CancelCampaignResponse:
    """Cancel every non-terminal job in a campaign."""
    try:
        n = await train_jobs.cancel_campaign(campaign_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CancelCampaignResponse(cancelled=n, campaign_id=campaign_id)
