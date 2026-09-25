"""C1: an in-process background job wrapping
:func:`src.services.curation.probe_predictions.run_probe_inference`, so
``POST /curation/probe/run`` doesn't block the request on a
potentially-long CPU/GPU inference pass over the whole items index.

Deliberately NOT the subprocess/state-file job runner
:mod:`src.services.curation.autolabel.job` uses (that machinery is
sized for a separate long-lived worker process with its own heartbeat
and crash recovery). A probe pass reuses the same API process and GPU
the training job already runs in, so a single ``asyncio.Task`` plus a
module-level "one job at a time" guard is enough: this endpoint's whole
reason to exist is convenience over the existing
``scripts/curation/run_probe.py`` operator driver, not a new execution
model.

GPU: this module never decides whether to use one -- the caller passes
``gpu`` (a ``cuda_visible_devices`` string) or ``None`` for CPU. When a
GPU is requested it is claimed through the exact same
:mod:`src.services.training.gpu_arbiter` path ``POST /train/start`` uses
(:func:`claim_gpus_for_training` / :func:`release_gpus_after_training`)
so a probe run pauses/stops the same GPU-resident services a training
run would, and is released the same way. A claim failure
(:class:`GpuArbiterStopFailedError`) is a hard error -- it propagates to
the caller (mapped to 409) rather than silently running the probe
unclaimed next to whatever else is on that GPU.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)


class ProbeJobBusyError(Exception):
    """A probe job is already running; only one may run at a time."""


@dataclass
class _ProbeJobState:
    job_id: str | None = None
    status: str = 'idle'  # idle | running | completed | failed | cancelled
    train_job_id: str | None = None
    model_path: str | None = None
    gpu: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_count: int | None = None
    error: str | None = None


_state = _ProbeJobState()
_task: asyncio.Task[Any] | None = None
_lock = asyncio.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def get_status() -> dict[str, Any]:
    """The current/last probe job's state (BA-style poll endpoint)."""
    return {
        'job_id': _state.job_id,
        'status': _state.status,
        'train_job_id': _state.train_job_id,
        'model_path': _state.model_path,
        'gpu': _state.gpu,
        'started_at': _state.started_at,
        'finished_at': _state.finished_at,
        'updated_count': _state.updated_count,
        'error': _state.error,
    }


async def _run(
    job_id: str,
    train_job_id: str,
    model_path: Path,
    opensearch: Any,
    *,
    config: Any,
    gpu: str | None,
    architecture: str,
    resume: bool,
) -> None:
    from src.services.curation.probe_predictions import run_probe_inference

    try:
        updated = await run_probe_inference(
            model_path,
            opensearch,
            config=config,
            model_version=job_id,
            architecture=architecture,
            resume=resume,
        )
        _state.status = 'completed'
        _state.updated_count = updated
    except asyncio.CancelledError:
        _state.status = 'cancelled'
        raise
    except Exception as exc:
        logger.error('probe_job_failed', job_id=job_id, train_job_id=train_job_id, error=str(exc))
        _state.status = 'failed'
        _state.error = str(exc)
    finally:
        _state.finished_at = _now_iso()
        if gpu is not None:
            from src.services.training.gpu_arbiter import release_gpus_after_training

            try:
                await release_gpus_after_training(gpu)
            except Exception as exc:
                logger.error('probe_job_gpu_release_failed', job_id=job_id, error=str(exc))


async def start_probe_job(
    job_id: str,
    train_job_id: str,
    model_path: Path,
    opensearch: Any,
    *,
    config: Any = None,
    gpu: str | None = None,
    architecture: str = 'yolo11',
    resume: bool = False,
) -> dict[str, Any]:
    """Start a probe job. Raises :class:`ProbeJobBusyError` if one is
    already running (409 at the router). If ``gpu`` is set, claims it via
    the training GPU arbiter first -- a
    :class:`~src.services.training.gpu_arbiter.GpuArbiterStopFailedError`
    propagates (never silently ignored)."""
    global _task  # noqa: PLW0603 - single-job-at-a-time module state
    async with _lock:
        if _state.status == 'running':
            msg = f'a probe job is already running: {_state.job_id!r}'
            raise ProbeJobBusyError(msg)

        if gpu is not None:
            from src.services.training.gpu_arbiter import claim_gpus_for_training

            await claim_gpus_for_training(gpu)  # GpuArbiterStopFailedError propagates -- hard fail

        _state.job_id = job_id
        _state.status = 'running'
        _state.train_job_id = train_job_id
        _state.model_path = str(model_path)
        _state.gpu = gpu
        _state.started_at = _now_iso()
        _state.finished_at = None
        _state.updated_count = None
        _state.error = None

        _task = asyncio.create_task(
            _run(
                job_id,
                train_job_id,
                model_path,
                opensearch,
                config=config,
                gpu=gpu,
                architecture=architecture,
                resume=resume,
            )
        )
    return get_status()


def cancel_probe_job() -> bool:
    """Request cancellation of the active probe job. Best-effort: the
    running :func:`~src.services.curation.probe_predictions.run_probe_inference`
    call is cancelled at its next ``await`` point, not mid-instruction."""
    if _task is not None and not _task.done() and _state.status == 'running':
        _task.cancel()
        return True
    return False


def _reset_for_tests() -> None:
    global _task  # noqa: PLW0603
    _task = None
    _state.job_id = None
    _state.status = 'idle'
    _state.train_job_id = None
    _state.model_path = None
    _state.gpu = None
    _state.started_at = None
    _state.finished_at = None
    _state.updated_count = None
    _state.error = None


__all__ = [
    'ProbeJobBusyError',
    'cancel_probe_job',
    'get_status',
    'start_probe_job',
]
