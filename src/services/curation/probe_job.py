"""The queued probe job: :func:`run_probe_inference` over the whole items index.

``POST /probe/run`` queues a ``probe`` job (one at a time per project); a Celery worker runs
:func:`run_probe`. A requested GPU is claimed through the same
:mod:`src.services.training.gpu_arbiter` path ``POST /train/start`` uses, inside the worker,
and released when the run ends; a claim failure fails the job (never a silent unclaimed run).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    import uuid
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.schemas.jobs import JobView
    from src.tasks.runtime.context import JobContext

logger = get_logger(__name__)

KIND = 'probe'


async def start(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    train_job_id: str,
    model_path: Path,
    gpu: str | None,
    architecture: str,
    resume: bool,
) -> JobRow:
    """Queue the probe pass. ``ValueError`` for an unknown ``architecture`` (422 at the router),
    409 ``job_busy`` while another probe of the project is active."""
    from src.services.curation.probe_models import PROBE_ARCHITECTURES

    if architecture not in PROBE_ARCHITECTURES:
        msg = f'unknown probe architecture {architecture!r} (expected one of {PROBE_ARCHITECTURES})'
        raise ValueError(msg)
    started = await start_job(
        session,
        KIND,
        project,
        params={
            'train_job_id': train_job_id,
            'model_path': str(model_path),
            'gpu': gpu,
            'architecture': architecture,
            'resume': resume,
        },
        requested_by='api',
    )
    return started.job


def status_dict(view: JobView | None) -> dict[str, Any]:
    """The probe status wire (``ProbeStatusResponse`` minus the threshold) of a job row."""
    if view is None:
        return {'status': 'idle'}
    params = view.params or {}
    return {
        'job_id': view.id,
        'status': view.status,
        'train_job_id': params.get('train_job_id'),
        'model_path': params.get('model_path'),
        'gpu': params.get('gpu'),
        'started_at': None if view.started_at is None else view.started_at.isoformat(),
        'finished_at': None if view.finished_at is None else view.finished_at.isoformat(),
        'updated_count': (view.result or {}).get('updated_count'),
        'error': view.error_message,
    }


async def run_probe(ctx: JobContext, opensearch: Any) -> dict[str, Any]:
    """Execute the job: the existing probe inference, checking for a cancel once per page."""
    from src.services.curation.probe_predictions import run_probe_inference

    params = ctx.params
    gpu = params.get('gpu')
    if gpu is not None:
        from src.services.training.gpu_arbiter import claim_gpus_for_training

        await claim_gpus_for_training(gpu)  # GpuArbiterStopFailedError fails the job
    try:
        from pathlib import Path

        updated = await run_probe_inference(
            Path(params['model_path']),
            opensearch,
            model_version=params['train_job_id'],
            architecture=params['architecture'],
            resume=bool(params['resume']),
            should_cancel=ctx.cancelled,
        )
    finally:
        if gpu is not None:
            from src.services.training.gpu_arbiter import release_gpus_after_training

            try:
                await release_gpus_after_training(gpu)
            except Exception as exc:
                logger.error('probe_job_gpu_release_failed', job_id=str(ctx.job_id), error=str(exc))
    return {'updated_count': updated, 'cancelled': ctx.cancelled()}


__all__ = ['KIND', 'run_probe', 'start', 'status_dict']
