"""C1: run the active-learning probe from a finished training run.

``POST /probe/run`` resolves the given training ``job_id``'s exported
weights (the same ``checkpoint_path`` ``POST /train/promote`` reads) and
runs :func:`src.services.curation.probe_predictions.run_probe_inference`
as a background task -- ``scripts/curation/run_probe.py`` stays the
operator-driven equivalent for an ad-hoc/manual checkpoint; this route
is the API path for "score the pool with the run I just trained" without
a shell into the container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from src.routers.curation._common import OpenSearchDep, router


class ProbeRunRequest(BaseModel):
    job_id: str
    gpu: str | None = None
    architecture: str = 'yolo11'
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


@router.post('/probe/run', response_model=ProbeStatusResponse)
async def probe_run(payload: ProbeRunRequest, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Start a probe pass from a finished training run's export.

    ``409`` when: the referenced training job isn't finished or has no
    exported checkpoint, or a probe job is already running. GPU claim
    (when ``gpu`` is given) goes through the same arbiter
    ``POST /train/start`` uses; a claim failure is also ``409`` (never
    silent -- see ``src.services.curation.probe_job``).
    """
    from src.services.curation.probe_job import ProbeJobBusyError, start_probe_job
    from src.services.training.gpu_arbiter import GpuArbiterStopFailedError

    weights = await _resolve_probe_weights(payload.job_id)
    try:
        return await start_probe_job(
            payload.job_id,
            payload.job_id,
            weights,
            opensearch,
            gpu=payload.gpu,
            architecture=payload.architecture,
            resume=payload.resume,
        )
    except ProbeJobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GpuArbiterStopFailedError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f'cannot claim GPU {payload.gpu!r} for the probe run: a configured '
                f'GPU-resident container could not be stopped ({exc})'
            ),
        ) from exc


@router.get('/probe/status', response_model=ProbeStatusResponse)
async def probe_status() -> dict[str, Any]:
    """Poll the current/last probe job."""
    from src.services.curation.probe_job import get_status

    return get_status()


@router.post('/probe/cancel')
async def probe_cancel() -> dict[str, Any]:
    """Best-effort cancel of the active probe job."""
    from src.services.curation.probe_job import cancel_probe_job, get_status

    cancelled = cancel_probe_job()
    return {'cancelled': cancelled, **get_status()}
