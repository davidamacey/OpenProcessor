"""A file-backed job runner wrapping
:func:`src.services.curation.probe_predictions.run_probe_inference`, so
``POST /curation/probe/run`` doesn't block the request on a
potentially-long CPU/GPU inference pass over the whole items index.

Runs in-process (an ``asyncio.Task`` inside the yolo-api process itself,
sharing the same GPU-claim path ``POST /train/start`` uses) -- unlike
:mod:`src.services.curation.autolabel.job`, there is no separate probe
worker container, so this mirrors :mod:`src.services.curation.item_scores.job`
/ :mod:`src.services.curation.embedding_viz`'s simpler in-process
state.json/heartbeat/cancel.flag convention instead of ``autolabel.job``'s
trigger-file dispatch to a long-lived worker.

**Multi-worker correctness (2026-09-25 fix).** ``yolo-api`` runs under
``uvicorn --workers=8`` -- eight separate OS processes. The previous
module-level ``_ProbeJobState`` dataclass plus ``_task`` handle were
invisible across workers: ``POST /probe/run`` landing on worker A and a
subsequent ``GET /probe/status`` landing on worker B always read
``idle`` on B regardless of what A was doing (verified live: 8 consecutive
status polls after a run start all read ``idle``), a cancel request could
never reach the job unless it happened to land back on worker A, and a
second ``POST /probe/run`` landing on a different worker could pass the
busy check and start a parallel run against the same items index. Fixed
by moving to the same on-disk, atomically-written state.json + heartbeat
+ cancel.flag convention ``item_scores.job``/``embedding_viz`` already
use, resolved fresh per call from ``OP_PROBE_JOBS_DIR`` (default
``/jobs/probe`` -- the shared ``/jobs`` volume every yolo-api worker
process mounts, consistent with ``item_scores``' ``/jobs/scores`` and
``embedding_viz``'s ``/jobs/viz``).

Singleton start is additionally guarded by an ``fcntl.flock`` lock file
(:mod:`src.services.curation.job_lock`) so the check-is-busy +
write-'running' sequence is atomic across processes, not just within one
-- see that module's docstring. The same cross-process gap existed in
``item_scores.job.start_job`` (a bare ``_is_busy()`` check with no lock at
all) and is fixed there too, in the same change that introduced this
module's rewrite.

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
import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.logging import get_logger


logger = get_logger(__name__)

_HEARTBEAT_STALE_S = 30.0

# Heartbeat tick cadence while a run is active. A full pass over the
# items index can run well past _HEARTBEAT_STALE_S between individual
# scroll pages (page_size items x model inference each), so a dedicated
# ticker task -- not just a touch at page boundaries -- keeps the
# heartbeat fresh throughout (mirrors embedding_viz._heartbeat_ticker).
_HEARTBEAT_TICK_S = 10.0

# Reference to the scheduled background task -- asyncio only holds a weak
# reference internally, so an unreferenced task can be garbage-collected
# mid-run. Per-process only (not consulted for cross-process state; that
# all lives in the state file) -- kept for parity with the other job
# modules' identical comment.
_active_task: asyncio.Task[None] | None = None


class ProbeJobBusyError(Exception):
    """A probe job is already running; only one may run at a time."""


def _jobs_dir() -> Path:
    """Resolved fresh each call so tests can override via monkeypatch
    (same convention as ``item_scores.job._state_dir`` /
    ``embedding_viz._jobs_dir``)."""
    return Path(os.environ.get('OP_PROBE_JOBS_DIR', '/jobs/probe'))


def _state_file() -> Path:
    return _jobs_dir() / 'state.json'


def _cancel_flag() -> Path:
    return _jobs_dir() / 'cancel.flag'


def _heartbeat_file() -> Path:
    return _jobs_dir() / 'heartbeat'


def _lock_file() -> Path:
    return _jobs_dir() / 'start.lock'


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ensure_dir() -> None:
    _jobs_dir().mkdir(parents=True, exist_ok=True)


def _atomic_write(state: _ProbeJobState) -> None:
    _ensure_dir()
    tmp = _state_file().with_suffix('.tmp')
    tmp.write_text(json.dumps(state.to_dict(), default=str))
    tmp.replace(_state_file())


def _read_state() -> _ProbeJobState:
    try:
        raw = json.loads(_state_file().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return _ProbeJobState()
    state = _ProbeJobState()
    for k, v in raw.items():
        if hasattr(state, k):
            setattr(state, k, v)
    return state


def _touch_heartbeat() -> None:
    _ensure_dir()
    _heartbeat_file().touch()


def _heartbeat_age() -> float | None:
    try:
        mtime = _heartbeat_file().stat().st_mtime
    except (FileNotFoundError, OSError):
        return None
    return max(0.0, time.time() - mtime)


async def _heartbeat_ticker() -> None:
    """Keep the heartbeat fresh across the whole run, not just at page
    boundaries -- same reasoning as ``embedding_viz._heartbeat_ticker``."""
    while True:
        await asyncio.sleep(_HEARTBEAT_TICK_S)
        _touch_heartbeat()


def is_cancelled() -> bool:
    """Cheap, file-existence check -- passed as ``run_probe_inference``'s
    ``should_cancel`` callable, checked once per scroll page."""
    return _cancel_flag().exists()


def _is_busy() -> bool:
    state = _read_state()
    if state.status != 'running':
        return False
    age = _heartbeat_age()
    # No heartbeat yet just means the task hasn't ticked once -- still busy.
    return age is None or age <= _HEARTBEAT_STALE_S


def reconcile_orphaned_jobs() -> bool:
    """Startup-only repair: see :mod:`src.services.curation.job_reconcile`.

    Called from ``src.main``'s lifespan before any request is served, so
    nothing in this process can legitimately hold ``status='running'``
    yet -- a leftover 'running' state.json is necessarily orphaned by a
    prior process. Returns True if the file was rewritten.
    """
    from src.services.curation.job_reconcile import reconcile_stale_running

    return reconcile_stale_running(
        _state_file(),
        _heartbeat_file(),
        stale_s=_HEARTBEAT_STALE_S,
        error_prefix='probe job',
    )


def get_status() -> dict[str, Any]:
    """The current/last probe job's state (BA-style poll endpoint), with
    stale-heartbeat repair mirroring ``item_scores.job.get_state``'s
    liveness contract: a worker process that died mid-run (or was killed)
    must not leave the job 'running' forever for every other worker's
    status poll -- the next poll (from any process) that notices the
    heartbeat is stale rewrites the state to 'failed' itself."""
    state = _read_state()
    if state.status == 'running':
        age = _heartbeat_age()
        if age is not None and age > _HEARTBEAT_STALE_S:
            state.status = 'failed'
            state.error = state.error or f'probe job heartbeat stale ({age:.1f}s ago)'
            state.finished_at = state.finished_at or _now_iso()
            _atomic_write(state)
    return state.to_dict()


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

    state = _read_state()
    if state.job_id != job_id:
        # Superseded by a newer job (shouldn't happen -- start_probe_job
        # is a cross-process singleton gate) -- bail rather than clobber
        # someone else's run.
        return
    _touch_heartbeat()
    ticker = asyncio.create_task(_heartbeat_ticker())
    try:
        updated = await run_probe_inference(
            model_path,
            opensearch,
            config=config,
            model_version=job_id,
            architecture=architecture,
            resume=resume,
            should_cancel=is_cancelled,
        )
        state = _read_state()
        if is_cancelled():
            state.status = 'cancelled'
        else:
            state.status = 'completed'
            state.updated_count = updated
        state.finished_at = _now_iso()
        _atomic_write(state)
    except asyncio.CancelledError:
        state = _read_state()
        state.status = 'cancelled'
        state.finished_at = _now_iso()
        _atomic_write(state)
        raise
    except Exception as exc:
        logger.error('probe_job_failed', job_id=job_id, train_job_id=train_job_id, error=str(exc))
        state = _read_state()
        state.status = 'failed'
        state.error = str(exc)
        state.finished_at = _now_iso()
        _atomic_write(state)
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
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
    already running anywhere in the fleet (409 at the router) -- the
    check-and-claim is atomic across processes, not just within one (see
    module docstring / :mod:`src.services.curation.job_lock`). If ``gpu``
    is set, claims it via the training GPU arbiter first -- a
    :class:`~src.services.training.gpu_arbiter.GpuArbiterStopFailedError`
    propagates (never silently ignored)."""
    from src.services.curation.job_lock import exclusive_start_lock

    global _active_task  # noqa: PLW0603 - singleton task handle, mirrors item_scores.job

    with exclusive_start_lock(_lock_file()) as acquired:
        if not acquired:
            msg = 'a probe job start is already in progress on another worker'
            raise ProbeJobBusyError(msg)

        if _is_busy():
            busy_state = _read_state()
            msg = f'a probe job is already running: {busy_state.job_id!r}'
            raise ProbeJobBusyError(msg)

        if gpu is not None:
            from src.services.training.gpu_arbiter import claim_gpus_for_training

            await claim_gpus_for_training(gpu)  # GpuArbiterStopFailedError propagates -- hard fail

        _ensure_dir()
        with contextlib.suppress(FileNotFoundError):
            _cancel_flag().unlink()
        with contextlib.suppress(FileNotFoundError):
            _heartbeat_file().unlink()

        state = _ProbeJobState(
            job_id=job_id,
            status='running',
            train_job_id=train_job_id,
            model_path=str(model_path),
            gpu=gpu,
            started_at=_now_iso(),
        )
        _atomic_write(state)

        _active_task = asyncio.create_task(
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
    return state.to_dict()


def cancel_probe_job() -> bool:
    """Touch the cross-process cancel flag if a job appears to be
    running. Best-effort and cooperative: the running
    :func:`~src.services.curation.probe_predictions.run_probe_inference`
    call notices at its next per-page ``should_cancel`` check, not
    mid-page -- and (unlike the old in-process ``asyncio.Task.cancel()``)
    this reaches the job regardless of which worker process is actually
    running it, since every process shares the same cancel.flag file."""
    if not _is_busy():
        return False
    _ensure_dir()
    _cancel_flag().touch()
    return True


def _reset_for_tests() -> None:
    """Kept for API-compat with any caller expecting it, but a file-backed
    job has no meaningful module-level state left to reset -- tests
    should instead point ``OP_PROBE_JOBS_DIR`` at a fresh ``tmp_path``."""
    global _active_task  # noqa: PLW0603
    _active_task = None
    with contextlib.suppress(FileNotFoundError):
        _state_file().unlink()
    with contextlib.suppress(FileNotFoundError):
        _cancel_flag().unlink()
    with contextlib.suppress(FileNotFoundError):
        _heartbeat_file().unlink()
    with contextlib.suppress(FileNotFoundError):
        _lock_file().unlink()


__all__ = [
    'ProbeJobBusyError',
    'cancel_probe_job',
    'get_status',
    'is_cancelled',
    'reconcile_orphaned_jobs',
    'start_probe_job',
]
