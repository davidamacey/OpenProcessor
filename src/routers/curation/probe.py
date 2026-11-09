"""Run the active-learning probe from a finished training run.

``POST /probe/run`` resolves the given training ``job_id``'s exported
weights (the same ``checkpoint_path`` ``POST /train/promote`` reads) and
runs :func:`src.services.curation.probe_predictions.run_probe_inference`
as a queued job -- ``scripts/curation/run_probe.py`` stays the
operator-driven equivalent for an ad-hoc/manual checkpoint; this route
is the API path for "score the pool with the run I just trained" without
a shell into the container.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.domain.model_families import PROBE_ARCHITECTURES
from src.routers.curation._common import router


class ProbeRunRequest(BaseModel):
    job_id: str
    gpu: str | None = None
    # yolo26 is the only trained family (G-22); the rest of
    # PROBE_ARCHITECTURES remains valid for older checkpoints or a
    # second-family reuse. Validated against PROBE_ARCHITECTURES in
    # probe_job.start (422 on mismatch) rather than here, so the error
    # message stays in one place.
    architecture: str = Field(default='yolo26', description=f'One of {PROBE_ARCHITECTURES}.')
    resume: bool = False


class ProbeStatusResponse(BaseModel):
    job_id: str | None = None
    status: str
    train_job_id: str | None = None
    model_path: str | None = None
    gpu: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_count: int | None = None
    error: str | None = None
    # Read-only echo of CurationConfig.probe_actionable_min_confidence --
    # the floor `probe_pred_confidence` must clear (alongside in-scope +
    # disagreeing) for an item's `probe_actionable` to be true. Served here
    # (not just baked into the wire) so the UI can explain "model unsure"
    # without hardcoding the threshold. Present on every poll, independent
    # of job state.
    actionable_min_confidence: float | None = None


async def _resolve_probe_weights(train_job_id: str) -> Path:
    """The finished run's exported checkpoint, or ``409`` when missing /
    the run isn't finished yet.

    Resolved the same way ``POST /train/promote/{job_id}`` does --
    ``TrainJobStatus.checkpoint_path`` (see
    ``src/services/training/triton_promote.py``'s promote path) -- so a
    probe run always uses the exact artifact promote would.
    """
    from src.services.training.jobs import read_status

    status = await read_status(train_job_id)
    if status is None:
        raise HTTPException(status_code=409, detail=f'unknown training job {train_job_id!r}')
    if status.state != 'finished':
        raise HTTPException(
            status_code=409,
            detail=f'training job {train_job_id!r} is not finished (state={status.state!r})',
        )
    if not status.checkpoint_path:
        raise HTTPException(
            status_code=409,
            detail=f'training job {train_job_id!r} has no exported checkpoint_path',
        )
    weights = Path(status.checkpoint_path)
    if not weights.is_file():
        raise HTTPException(
            status_code=409,
            detail=f"training job {train_job_id!r}'s checkpoint is missing on disk: {weights}",
        )
    return weights


@router.post('/probe/run', response_model=ProbeStatusResponse, status_code=202)
async def probe_run(payload: ProbeRunRequest, session: DbSessionDep) -> dict[str, Any]:
    """Queue a probe pass from a finished training run's export (202 with the job).

    ``409`` when the referenced training job isn't finished or has no exported checkpoint, and
    ``409 job_busy`` while a probe job is active. ``422`` when ``architecture`` is outside
    :data:`~src.domain.model_families.PROBE_ARCHITECTURES`. A requested ``gpu`` is
    claimed by the worker through the arbiter ``POST /train/start`` uses; a claim failure fails
    the job (see ``src.services.curation.probe_job``).
    """
    from src.services.curation import probe_job

    weights = await _resolve_probe_weights(payload.job_id)
    try:
        await probe_job.start(
            session,
            current_project_id(),
            train_job_id=payload.job_id,
            model_path=weights,
            gpu=payload.gpu,
            architecture=payload.architecture,
            resume=payload.resume,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await _status(session)


async def _status(session: Any) -> dict[str, Any]:
    from src.config import get_curation_config
    from src.services.curation.latest_job import latest_job
    from src.services.curation.probe_job import status_dict

    return {
        **status_dict(await latest_job(session, 'probe')),
        'actionable_min_confidence': get_curation_config().probe_actionable_min_confidence,
    }


@router.get('/probe/status', response_model=ProbeStatusResponse)
async def probe_status(session: DbSessionDep) -> dict[str, Any]:
    """Poll the current/last probe job, plus the read-only
    ``actionable_min_confidence`` threshold gating the item wire's
    ``probe_actionable`` field."""
    return await _status(session)


@router.post('/probe/cancel')
async def probe_cancel(session: DbSessionDep) -> dict[str, Any]:
    """Cancel the active probe job, if there is one."""
    from src.services.curation.latest_job import ACTIVE_STATUSES, latest_job
    from src.services.jobs.service import cancel_job

    current = await latest_job(session, 'probe')
    cancelled = current is not None and current.status in ACTIVE_STATUSES
    if cancelled:
        assert current is not None
        await cancel_job(session, uuid.UUID(current.id), current_project_id(), requested_by='api')
    return {'cancelled': cancelled, **await _status(session)}
