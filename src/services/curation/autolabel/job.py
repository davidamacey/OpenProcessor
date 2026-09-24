"""File-backed dispatcher for ``/legacy/pipeline/auto_label``.

The actual pipeline now runs in a **dedicated long-lived worker
container** (``curation-auto-label-worker``, entrypoint
``scripts/curation/auto_label_worker.py``). yolo-api just drops a
trigger JSON file on the shared ``/jobs`` volume and returns 202.
The worker picks it up, runs the pipeline, and writes back to the
same state.json the SSE endpoint already tails.

This replaces a previous per-request ``subprocess.Popen`` model that
had two failure modes the worker container fixes:

* The subprocess died silently in the ~2 ms window between
  ``subprocess.Popen`` returning and the subprocess writing
  ``running.lock``. The status poll would see no lock + state='running'
  and stamp 'subprocess vanished before writing terminal state'.
* ``/proc/<pid>/stat`` liveness checks were meaningless across
  container boundaries anyway, so the lock contract was inherently
  per-host-PID-namespace.

The worker writes a heartbeat file every ~5 s while a run is active;
``_is_busy()`` uses the heartbeat mtime (cross-container compatible)
instead of the old /proc inspection. Stale heartbeat (> 15 s) means
the worker died — same repair path stamps 'vanished' in state.json.

State files (on the shared ``/jobs`` volume):

* ``/jobs/auto_label/trigger.json`` — written by ``start_job``;
  consumed (unlinked) by the worker. Format
  ``{job_id, pipeline, args}``.
* ``/jobs/auto_label/state.json`` — single source of truth for status.
  Worker writes at every progress checkpoint, atomic via temp+rename.
* ``/jobs/auto_label/heartbeat`` — worker touches every 5 s while a
  run is active. Liveness signal for the API.
* ``/jobs/auto_label/cancel.flag`` — empty file. Worker polls it at
  progress checkpoints and raises ``CancelledError``. No SIGTERM
  fallback because the worker is in a different container — operators
  can ``docker compose stop curation-auto-label-worker`` for an
  immediate hard cancel if they need one.
* ``/jobs/auto_label/running.lock`` — JSON ``{pid, owner, ...}``
  written by the worker. Kept for back-compat with tools that grep
  for it; the API trusts the heartbeat, not the pid.

Public API:

* :func:`start_job` — write the trigger file, return a snapshot.
* :func:`get_state` — read state.json (with stale-heartbeat repair).
* :func:`cancel_job` — touch the cancel flag.
* :func:`auto_label_changed_event` — asyncio.Event signalled whenever
  ``state.json`` is rewritten (driven by the inotify watcher below).
* :func:`watch_state_file` — lifespan task that fires the event.
* :func:`shutdown_active_run` — no-op kept for API compatibility;
  the worker container has its own restart policy.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STAGES: tuple[str, ...] = (
    'cluster_id_normalize',
    'cluster_residuals',
    'auto_promote',
    'vlm',
    'finalize',
)


_STATE_DIR = Path(os.environ.get('OP_AUTO_LABEL_STATE_DIR', '/jobs/auto_label'))
_STATE_FILE = _STATE_DIR / 'state.json'
_CANCEL_FLAG = _STATE_DIR / 'cancel.flag'
_RUNNING_LOCK = _STATE_DIR / 'running.lock'
_EXIT_CODE_FILE = _STATE_DIR / 'exit_code'
_TRIGGER_FILE = _STATE_DIR / 'trigger.json'
_HEARTBEAT_FILE = _STATE_DIR / 'heartbeat'

# How stale the worker's heartbeat must be before we declare the worker
# dead and stamp 'vanished' on a 'running' state. The worker touches
# heartbeat every 5 s; 30 s leaves room for a long blocking call inside
# a stage without false-positives.
_HEARTBEAT_STALE_S = 30.0

# Module-level asyncio Event signalled whenever state.json is rewritten.
# The SSE endpoint awaits this; an inotify watcher started in the FastAPI
# lifespan does the signalling. Per-uvicorn-worker; each worker watches
# the same on-disk file.
_changed_event: asyncio.Event | None = None


@dataclass
class _JobState:
    job_id: str = ''
    status: str = 'idle'  # 'idle' | 'running' | 'completed' | 'failed' | 'cancelled'
    stage: str = ''
    processed: int = 0
    total: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str | None = None
    error_detail: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)
    pipeline: str = ''  # 'module:function' import path of the pipeline_fn
    # GPU clustering telemetry — None on CPU-only runs.
    backend: str | None = None  # 'gpu' | 'cpu'
    backend_detail: str | None = None
    free_vram_mb: int | None = None
    peak_vram_mb: int | None = None
    # Wall-clock seconds spent in each named stage. Filled in by
    # _Progress as the pipeline advances; rendered as a stage-timings
    # expander in the labeler's run summary.
    stage_durations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        elapsed = (time.time() - self.started_at) if self.status == 'running' else 0.0
        if self.status == 'running' and self.processed > 0 and self.total > self.processed:
            per_item = elapsed / self.processed
            d['eta_seconds'] = round(per_item * (self.total - self.processed), 1)
        else:
            d['eta_seconds'] = None
        d['elapsed_seconds'] = round(elapsed, 1) if elapsed else 0.0
        return d


def _ensure_dir() -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write(payload: dict[str, Any]) -> None:
    """Write state.json via temp + rename so readers never see a partial file."""
    _ensure_dir()
    tmp = _STATE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(_STATE_FILE)


def _read_state() -> _JobState:
    """Load the on-disk state. Returns a default idle state if missing."""
    try:
        raw = json.loads(_STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return _JobState()
    state = _JobState()
    for k, v in raw.items():
        if k in {'eta_seconds', 'elapsed_seconds'}:
            continue  # derived in to_dict
        if hasattr(state, k):
            setattr(state, k, v)
    return state


def _heartbeat_age() -> float | None:
    """Seconds since the worker last touched HEARTBEAT_FILE.

    Returns None if the file does not exist (worker is idle / no run in
    flight). Callers compare against ``_HEARTBEAT_STALE_S`` to decide if
    a 'running' state should be repaired to 'failed'.
    """
    try:
        mtime = _HEARTBEAT_FILE.stat().st_mtime
    except (FileNotFoundError, PermissionError, OSError):
        return None
    return max(0.0, time.time() - mtime)


def _is_busy() -> bool:
    """Report whether the auto-label worker has a live run in flight.

    Cross-container compatible: uses the heartbeat-file mtime instead of
    ``/proc/<pid>/stat`` (which is namespaced and so was meaningless
    when the API and the worker were in different containers). The
    worker container touches HEARTBEAT_FILE every 5 s; if the mtime is
    less than ``_HEARTBEAT_STALE_S`` old we consider the run alive.

    A pending trigger that hasn't been picked up yet also counts as
    'busy' — a second :func:`start_job` while a trigger sits unread
    would otherwise stomp on it.
    """
    if _TRIGGER_FILE.exists():
        return True
    age = _heartbeat_age()
    if age is None:
        return False
    return age <= _HEARTBEAT_STALE_S


def _reap_stale_artifacts() -> None:
    """Clean up the running.lock / heartbeat / cancel.flag left behind
    when ``get_state`` concludes the worker died mid-run.

    Safe to call concurrently — uses suppressed FileNotFoundError.
    """
    with contextlib.suppress(FileNotFoundError):
        _RUNNING_LOCK.unlink()
    with contextlib.suppress(FileNotFoundError):
        _CANCEL_FLAG.unlink()
    with contextlib.suppress(FileNotFoundError):
        _HEARTBEAT_FILE.unlink()


def reconcile_orphaned_jobs() -> bool:
    """Startup-only repair, called from ``src.main``'s lifespan.

    Unlike the other three job modules in this package, the auto-label
    pipeline runs in a *separate* long-lived worker container
    (``curation-auto-label-worker``) with its own restart lifecycle —
    restarting yolo-api does not kill that worker, so most of the time
    there is nothing to reconcile here. This still matters for the case
    where yolo-api itself was down (and so never got to run this check)
    while the worker died mid-run: without this, nothing repairs
    ``state.json`` until the next status poll. A pending, not-yet-claimed
    trigger is left alone — the worker container may simply not have
    gotten to it yet, which is not an orphaned run.

    Uses ``'interrupted'`` (via the shared helper) rather than this
    module's own lazy ``get_state()`` repair status (``'failed'``) so the
    two repair paths are distinguishable; since this runs first in the
    lifespan, 'interrupted' is what a caller normally observes for a
    heartbeat-stale run recovered at startup.
    """
    if _TRIGGER_FILE.exists():
        return False
    from src.services.curation.job_reconcile import reconcile_stale_running

    return reconcile_stale_running(
        _STATE_FILE,
        _HEARTBEAT_FILE,
        stale_s=_HEARTBEAT_STALE_S,
        error_prefix='auto_label worker',
    )


def get_state() -> dict[str, Any]:
    """Read-only snapshot from the on-disk state file.

    Repair logic: if state says 'running' but the worker's heartbeat is
    stale (> ``_HEARTBEAT_STALE_S``) AND there's no pending trigger,
    the worker container died mid-run without writing terminal state.
    Stamp the file failed with a heartbeat-staleness reason so
    operators see an actionable message instead of stale 'running'.

    A 'queued' state with no heartbeat is fine — the trigger is sitting
    in the worker's poll cycle. Liveness only matters once a run is
    actually executing.
    """
    state = _read_state()
    if state.status == 'running':
        age = _heartbeat_age()
        if not _TRIGGER_FILE.exists() and (age is None or age > _HEARTBEAT_STALE_S):
            stale_msg = (
                f'auto_label worker heartbeat stale ({age:.1f}s ago)'
                if age is not None
                else 'auto_label worker heartbeat missing'
            )
            state.status = 'failed'
            state.error = state.error or stale_msg
            state.finished_at = state.finished_at or time.time()
            _atomic_write(asdict(state))
            _reap_stale_artifacts()
    return state.to_dict()


class _Progress:
    """Progress reporter used by the worker. Writes through to the
    on-disk state file at every advance. Importable so test doubles can
    swap in a fake."""

    def __init__(self, base_state: _JobState) -> None:
        self._state = base_state
        # Wall-clock when the current stage started. Used to fold the
        # elapsed time into ``stage_durations`` on the next start_stage()
        # so the labeler can render "AHC: 287s, UMAP: 22s" etc.
        self._stage_started_at: float = 0.0

    def _flush_current_stage_duration(self) -> None:
        """Fold the current stage's wall time into stage_durations."""
        if self._state.stage and self._stage_started_at > 0:
            dur = max(0.0, time.time() - self._stage_started_at)
            self._state.stage_durations[self._state.stage] = round(
                self._state.stage_durations.get(self._state.stage, 0.0) + dur,
                2,
            )

    def start_stage(self, stage: str, total: int = 0) -> None:
        if stage not in STAGES:
            raise ValueError(f'unknown stage: {stage}')
        self._flush_current_stage_duration()
        self._state.stage = stage
        self._state.processed = 0
        self._state.total = int(total)
        self._stage_started_at = time.time()
        _atomic_write(asdict(self._state))

    def advance(self, n: int = 1) -> None:
        self._state.processed = min(
            self._state.processed + int(n),
            self._state.total or self._state.processed + int(n),
        )
        _atomic_write(asdict(self._state))

    def update(self, processed: int, total: int | None = None) -> None:
        """Set absolute progress. Used by scroll-loop checkpoints and
        elapsed-tick patterns where ``advance`` doesn't fit.

        ``total=None`` leaves the existing total in place.
        """
        self._state.processed = max(0, int(processed))
        if total is not None:
            self._state.total = max(0, int(total))
        _atomic_write(asdict(self._state))

    def set_backend(
        self,
        name: str,
        detail: str,
        free_vram_mb: int | None = None,
    ) -> None:
        """Record which clustering backend the run is using.

        Called once at the start of ``cluster_residuals`` so the
        dashboard chip flips from "preparing…" to "gpu (cuml …)" or
        "cpu (sklearn …)".
        """
        self._state.backend = name
        self._state.backend_detail = detail
        self._state.free_vram_mb = free_vram_mb
        _atomic_write(asdict(self._state))

    def record_peak_vram(self, mb: int | None) -> None:
        """Keep the maximum VRAM observed during the run."""
        if mb is None:
            return
        current = self._state.peak_vram_mb or 0
        if mb > current:
            self._state.peak_vram_mb = int(mb)
            _atomic_write(asdict(self._state))

    def finalize(self) -> None:
        """Flush the final stage's duration. Called by the worker when
        the pipeline returns or raises (so partial runs still report
        their per-stage timings)."""
        self._flush_current_stage_duration()
        self._stage_started_at = 0.0
        _atomic_write(asdict(self._state))

    @property
    def cancelled(self) -> bool:
        return _CANCEL_FLAG.exists()

    def raise_if_cancelled(self) -> None:
        if _CANCEL_FLAG.exists():
            raise asyncio.CancelledError('auto_label run cancelled by operator')


async def with_elapsed_tick(
    progress: _Progress | None,
    coro: Any,
    *,
    interval: float = 5.0,
) -> Any:
    """Run ``coro`` while a sibling task ticks ``progress.update(elapsed_s)``.

    Stages whose underlying operation (e.g. OpenSearch
    ``update_by_query`` with ``wait_for_completion=True``) offers no
    intermediate callback would otherwise show ``processed=0/0`` for
    the whole stage. The tick pattern emits the elapsed-seconds counter
    so the labeler can render "12s elapsed" instead of an indeterminate
    bar. ``total`` stays 0 — the bar stays indeterminate, but the
    numeric counter advances so operators see the run is alive.
    """
    if progress is None:
        return await coro

    started = time.time()

    async def _tick() -> None:
        while True:
            progress.update(processed=int(time.time() - started), total=0)
            await asyncio.sleep(interval)

    task = asyncio.create_task(coro)
    ticker = asyncio.create_task(_tick())
    try:
        return await task
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker


PipelineFn = Callable[..., Any]


def _pipeline_import_path(fn: PipelineFn) -> str:
    """``module:qualname`` so the subprocess can re-import the function."""
    mod = getattr(fn, '__module__', None)
    name = getattr(fn, '__qualname__', None) or getattr(fn, '__name__', None)
    if not mod or not name:
        raise ValueError(f'cannot derive import path for {fn!r}')
    return f'{mod}:{name}'


def _serializable(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Strip non-JSON-serializable values from kwargs.

    The OpenSearch client and progress reporters aren't crossable —
    the subprocess constructs its own client; progress is wired
    locally."""
    return {
        k: v
        for k, v in kwargs.items()
        if isinstance(v, (str, int, float, bool, list, tuple, dict, type(None)))
    }


def start_job(pipeline_fn: PipelineFn, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Write a trigger file the worker container will pick up.

    Returns the initial 'queued' state snapshot. The worker flips
    status to 'running' once it claims the trigger (typically within
    POLL_INTERVAL_S = 1 s of the trigger landing).

    Raises ``RuntimeError`` if a run is already in flight; the caller
    surfaces this as HTTP 409.
    """
    _ensure_dir()
    if _is_busy():
        raise RuntimeError('auto_label run already in progress')

    # Clean up any leftover sentinels from a prior aborted run so
    # operators get a clean slate. (cancel.flag would otherwise short-
    # circuit the next run at its first stage boundary.)
    with contextlib.suppress(FileNotFoundError):
        _CANCEL_FLAG.unlink()
    with contextlib.suppress(FileNotFoundError):
        _EXIT_CODE_FILE.unlink()
    with contextlib.suppress(FileNotFoundError):
        _HEARTBEAT_FILE.unlink()

    pipeline_path = _pipeline_import_path(pipeline_fn)
    serializable_args = _serializable(kwargs)
    job_id = uuid.uuid4().hex

    # Initial state — 'queued' until the worker picks it up. Distinct
    # from 'running' so the dashboard can show "waiting on worker"
    # without faking liveness.
    state = _JobState(
        job_id=job_id,
        status='queued',
        stage='',
        started_at=time.time(),
        args=serializable_args,
        pipeline=pipeline_path,
    )
    _atomic_write(asdict(state))

    # Drop the trigger. The worker watches for this file every
    # POLL_INTERVAL_S and claims it atomically via unlink.
    trigger_tmp = _TRIGGER_FILE.with_suffix('.tmp')
    trigger_tmp.write_text(
        json.dumps(
            {
                'job_id': job_id,
                'pipeline': pipeline_path,
                'args': serializable_args,
                'enqueued_at': time.time(),
            }
        )
    )
    trigger_tmp.replace(_TRIGGER_FILE)
    return state.to_dict()


def cancel_job() -> bool:
    """Touch the cancel flag for the worker to pick up.

    The worker checks ``cancel.flag`` at every stage boundary via
    :class:`_Progress`. Since the worker is in a separate container we
    can no longer send SIGTERM directly; the cooperative checkpoint
    poll is the only cross-container channel. Operators who need a
    HARD cancel mid-stage can ``docker compose stop
    curation-auto-label-worker`` (the worker's signal handler writes
    'cancelled' state on the way down).

    Returns True if a run was active.
    """
    if not _is_busy():
        return False
    _ensure_dir()
    _CANCEL_FLAG.touch()
    return True


def auto_label_changed_event() -> asyncio.Event:
    """Module-level asyncio.Event signalled whenever state.json is rewritten.

    SSE handlers await this. The first caller creates it (lazy init) so
    importing this module from a non-async context still works.
    """
    global _changed_event  # noqa: PLW0603 - intentional module-level lazy singleton
    if _changed_event is None:
        _changed_event = asyncio.Event()
    return _changed_event


def _signal_changed() -> None:
    """Best-effort: bump the asyncio Event if it exists. Called by the
    inotify watcher whenever state.json mtime advances."""
    ev = _changed_event
    if ev is not None and not ev.is_set():
        ev.set()


async def watch_state_file() -> None:
    """Long-running task: signal :func:`auto_label_changed_event`
    whenever ``state.json`` is rewritten.

    Uses inotify so there's **zero CPU when nothing is changing** — the
    handler blocks on ``loop.add_reader`` until the kernel reports an
    event. Designed for the FastAPI lifespan: launch with
    ``asyncio.create_task(watch_state_file())`` and ``.cancel()`` it on
    shutdown.

    Falls back to a 1-second mtime poll if inotify isn't available (e.g.
    running on a filesystem that doesn't support it). The poll is
    bounded; production should always have inotify.
    """
    _ensure_dir()
    # Touch the file so inotify has something to watch even before the
    # first run.
    if not _STATE_FILE.exists():
        _atomic_write(asdict(_JobState()))

    auto_label_changed_event()  # ensure the Event is created

    try:
        import inotify_simple  # type: ignore[import-not-found]
    except ImportError:
        await _watch_state_file_poll()
        return

    loop = asyncio.get_running_loop()
    inotify = inotify_simple.INotify()
    # IN_CLOSE_WRITE catches the atomic-rename target; IN_MOVED_TO catches
    # the rename itself (we write to .tmp then replace).
    flags = (
        inotify_simple.flags.CLOSE_WRITE
        | inotify_simple.flags.MOVED_TO
        | inotify_simple.flags.CREATE
    )
    inotify.add_watch(str(_STATE_DIR), flags)

    fd_event = asyncio.Event()
    loop.add_reader(inotify.fd, fd_event.set)
    try:
        while True:
            await fd_event.wait()
            fd_event.clear()
            for ev in inotify.read(timeout=0):
                if ev.name == _STATE_FILE.name:
                    _signal_changed()
                    break
    except asyncio.CancelledError:
        raise
    finally:
        loop.remove_reader(inotify.fd)
        inotify.close()


async def _watch_state_file_poll() -> None:
    """Fallback path for systems without inotify. 1 Hz mtime poll."""
    last_mtime = 0.0
    while True:
        try:
            mtime = _STATE_FILE.stat().st_mtime
        except FileNotFoundError:
            mtime = 0.0
        if mtime != last_mtime:
            last_mtime = mtime
            _signal_changed()
        await asyncio.sleep(1.0)


async def shutdown_active_run(timeout: float = 10.0) -> None:  # noqa: ARG001 — kept for lifespan compat
    """No-op. Kept for FastAPI lifespan signature compatibility.

    The auto_label pipeline now runs in a separate
    ``curation-auto-label-worker`` container with its own lifecycle.
    Stopping yolo-api no longer needs to (and can no longer) reach
    across containers to cancel an in-flight run. Operators who want
    to halt a run during an API restart should ``docker compose stop
    curation-auto-label-worker`` first; the worker's signal handler
    flips state.json to 'cancelled' on its way down.
    """
    return
