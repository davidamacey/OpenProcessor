"""Background promote jobs (``POST /train/promote/{job_id}`` without ``wait=true``).

A promote copies the ONNX into the Triton repo, asks Triton to load it, then
pays the TensorRT engine build with throwaway inferences: 2-3 minutes, too
long for one proxied HTTP request. This runs the same work as an in-process
task and persists its phase under the project's train jobs dir
(``<train_jobs_dir>/promote/<promote_id>/``) with the shared
:class:`~src.services.curation.file_job.FileJob` convention: atomic
``state.json`` + heartbeat on the shared volume, so any API worker answers
the status poll, a restart leaves ``failed`` (never a stuck active phase),
and the per-run singleton claim is the same ``flock`` start lock the other
file-backed jobs use. Only the process that accepted the POST runs the task.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.curation.file_job import FileJob, heartbeat_ticker
from src.services.curation.job_lock import exclusive_start_lock


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

logger = get_logger(__name__)

# Order is the order a healthy promote moves through. ``building`` is the
# first warm-up inference (the TensorRT accelerator compiles the engine
# there, not in Triton's /load); ``warming`` is the max-batch pass.
PHASES = ('queued', 'exporting', 'loading', 'building', 'warming', 'done', 'failed')
ACTIVE = frozenset({'queued', 'exporting', 'loading', 'building', 'warming'})

_tasks: dict[str, asyncio.Task[None]] = {}


class PromoteJobConflictError(Exception):
    """A promote of the same run is already active under another triton_name."""

    def __init__(self, promote_id: str, triton_name: str) -> None:
        super().__init__(
            f'promote {promote_id} of this run is already active as {triton_name!r}; '
            'wait for it to finish'
        )
        self.promote_id = promote_id
        self.triton_name = triton_name


def jobs_root() -> Path:
    from src.services.training.job_files import _resolve_jobs_dir

    return _resolve_jobs_dir() / 'promote'


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _job(promote_id: str) -> FileJob:
    if not promote_id or '/' in promote_id or '\\' in promote_id or promote_id.startswith('.'):
        raise KeyError(promote_id)
    return FileJob(jobs_root() / promote_id)


def _dirs() -> list[Path]:
    root = jobs_root()
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def _active_for_run(run_job_id: str) -> tuple[str, dict[str, Any]] | None:
    for path in _dirs():
        job = FileJob(path)
        state = job.read()
        if state.get('job_id') == run_job_id and job.is_live(ACTIVE):
            return path.name, state
    return None


def reconcile_orphaned_jobs() -> bool:
    repaired = False
    for path in _dirs():
        repaired |= FileJob(path).reconcile(active_statuses=ACTIVE, error_prefix='promote job')
    return repaired


def read_job(promote_id: str) -> dict[str, Any] | None:
    """The wire-shaped state, or ``None``. A job whose process died reads
    ``failed`` (the file says ``interrupted``) with the reason in ``error``."""
    try:
        job = _job(promote_id)
    except KeyError:
        return None
    if not job.read():
        return None
    state = job.repair_if_stale(ACTIVE, error_prefix='promote job')
    return _wire(promote_id, state)


def latest_for_run(run_job_id: str) -> dict[str, Any] | None:
    """The most recently started promote of ``run_job_id`` (for train status)."""
    best: tuple[str, str] | None = None
    for path in _dirs():
        state = FileJob(path).read()
        if state.get('job_id') == run_job_id:
            key = (str(state.get('started_at') or ''), path.name)
            if best is None or key > best:
                best = key
    return read_job(best[1]) if best else None


def _wire(promote_id: str, state: dict[str, Any]) -> dict[str, Any]:
    status = state.get('status', 'failed')
    if status == 'interrupted':
        status = 'failed'
    return {
        'promote_id': promote_id,
        'job_id': state.get('job_id'),
        'triton_name': state.get('triton_name'),
        'status': status,
        'error': state.get('error'),
        'error_status': state.get('error_status'),
        'result': state.get('result'),
        'started_at': state.get('started_at'),
        'updated_at': state.get('updated_at'),
        'finished_at': state.get('finished_at')
        if isinstance(state.get('finished_at'), str | None)
        else datetime.fromtimestamp(state['finished_at'], UTC).isoformat(),
        'poll_after_s': 3 if status in ACTIVE else None,
    }


def claim(*, run_job_id: str, triton_name: str) -> tuple[FileJob, bool]:
    """Claim the per-run singleton. Returns ``(job, created)``: when a promote
    of this run is already live, that job and ``False``. Raises
    :class:`PromoteJobConflictError` when the live one targets another name."""
    root = jobs_root()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_start_lock(root / 'start.lock') as acquired:
        active = _active_for_run(run_job_id)
        if active is not None:
            promote_id, state = active
            if state.get('triton_name') != triton_name:
                raise PromoteJobConflictError(promote_id, str(state.get('triton_name')))
            return _job(promote_id), False
        if not acquired:
            # another process is mid-claim; its job is not visible yet
            raise PromoteJobConflictError('starting', triton_name)
        promote_id = f'pm_{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}'
        job = _job(promote_id)
        job.directory.mkdir(parents=True)
        job.clear_signals()
        job.write(
            {
                'status': 'queued',
                'job_id': run_job_id,
                'triton_name': triton_name,
                'started_at': _now(),
                'updated_at': _now(),
            }
        )
        job.touch_heartbeat()
    return job, True


def start(job: FileJob, work: Callable[[Callable[[str], None]], Awaitable[dict[str, Any]]]) -> None:
    """Schedule ``work`` in this process. ``work`` receives a phase callback
    and returns the JSON-able result; any exception fails the job (an
    ``HTTPException``'s status is kept in ``error_status``)."""
    _tasks[job.directory.name] = asyncio.create_task(_run(job, work))


_detached: set[asyncio.Task[Any]] = set()


async def run_detached(work: Awaitable[Any]) -> Any:
    """Await ``work`` so that the *caller* being cancelled does not cancel it.

    ``?wait=true`` runs the whole export, load and warm-up inside one request.
    A proxy that times out (nginx 504 at 120 s) drops the connection and the
    handler can be cancelled mid-load, leaving a model Triton has half loaded
    and not ready. The work runs as its own task and is shielded: an aborted
    client abandons the response, never the promote, so the model finishes
    loading and warming.
    """
    task = asyncio.ensure_future(work)
    _detached.add(task)

    def _done(t: asyncio.Task[Any]) -> None:
        _detached.discard(t)
        if not t.cancelled() and t.exception() is not None:
            logger.warning('promote_wait_abandoned_failed', error=str(t.exception()))

    task.add_done_callback(_done)
    return await asyncio.shield(task)


async def _run(
    job: FileJob, work: Callable[[Callable[[str], None]], Awaitable[dict[str, Any]]]
) -> None:
    def on_phase(phase: str) -> None:
        job.update(status=phase, updated_at=_now())
        job.touch_heartbeat()

    ticker = asyncio.create_task(heartbeat_ticker(job))
    try:
        on_phase('exporting')
        result = await work(on_phase)
        job.update(status='done', result=result, finished_at=_now(), updated_at=_now())
    except asyncio.CancelledError:
        job.update(status='failed', error='cancelled', finished_at=_now(), updated_at=_now())
        raise
    except Exception as exc:
        # HTTPException carries .status_code/.detail; PromoteError .status_code
        code = getattr(exc, 'status_code', None)
        detail = getattr(exc, 'detail', None)
        logger.error('promote_job_failed', job=job.directory.name, error=str(exc))
        job.update(
            status='failed',
            error=str(detail if detail is not None else exc),
            error_status=code,
            finished_at=_now(),
            updated_at=_now(),
        )
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
        _tasks.pop(job.directory.name, None)


__all__ = [
    'ACTIVE',
    'PHASES',
    'PromoteJobConflictError',
    'claim',
    'jobs_root',
    'latest_for_run',
    'read_job',
    'reconcile_orphaned_jobs',
    'run_detached',
    'start',
]
