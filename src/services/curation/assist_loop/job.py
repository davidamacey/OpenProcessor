"""Where a loop lives on disk, and the protocol between the API and the loop worker.

State lives under ``<OP_ASSIST_LOOP_JOBS_DIR>/projects/<slug>/<loop_id>/`` on the shared
``/jobs`` volume (:class:`~src.services.curation.file_job.FileJob` plus the loop's own
files, see :mod:`~src.services.curation.assist_loop.store`). It is a sibling of the assist
runs root, never inside it: that root treats every subdirectory as a run and prunes the
oldest.

The API creates a loop as ``queued`` and writes the project's ``trigger.json`` naming it;
the worker claims the trigger by unlinking it and runs the loop. A resume re-queues a
stopped loop the same way. One loop per project is live at a time (``queued`` or
``running`` with a fresh heartbeat); the check and the claim happen under one start lock.
A queued loop has no heartbeat yet, which counts as live: it waits for the worker as long
as the worker takes.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.env import env_path
from src.config.project_context import ensure_marked_dir, mark_project_dir, project_jobs_dir
from src.services.curation.file_job import FileJob
from src.services.curation.job_lock import exclusive_start_lock
from src.services.rejections import rejection
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from pathlib import Path

ACTIVE = frozenset({'queued', 'running'})
RESUMABLE = frozenset({'interrupted', 'paused', 'cancelled'})
TRIGGER_FILE = 'trigger.json'
PAUSE_FLAG = 'pause.flag'


def jobs_root() -> Path:
    """The bound project's loops dir (read per call so tests can ``monkeypatch.setenv``)."""
    return mark_project_dir(
        project_jobs_dir(env_path('OP_ASSIST_LOOP_JOBS_DIR', '/jobs/assist_loop'))
    )


def open_job(loop_id: str) -> FileJob:
    """The job of ``loop_id``; ids are minted here, so one that could name a path is unknown."""
    if not loop_id or '/' in loop_id or '\\' in loop_id or loop_id.startswith('.'):
        raise KeyError(loop_id)
    return FileJob(jobs_root() / loop_id)


def _loop_dirs() -> list[Path]:
    root = jobs_root()
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def live_loop_ids() -> list[str]:
    return [p.name for p in _loop_dirs() if FileJob(p).is_live(ACTIVE)]


def loop_live() -> bool:
    return bool(live_loop_ids())


def require_no_live_loop() -> None:
    """The refresh hold: nothing may retrain the clusters under a running loop."""
    live = live_loop_ids()
    if live:
        raise rejection(
            409,
            'assist_loop_running',
            f'auto-label loop {live[0]} is running; cancel it before retraining the clusters',
            loop_id=live[0],
        )


def reconcile_orphaned_jobs() -> bool:
    """Startup repair for the bound project: a loop a dead process left active becomes
    ``interrupted``."""
    repaired = False
    for path in _loop_dirs():
        repaired |= FileJob(path).reconcile(active_statuses=ACTIVE, error_prefix='assist loop')
    return repaired


def read_state(loop_id: str) -> dict[str, Any] | None:
    """The loop's state, with a loop whose heartbeat went stale repaired to ``interrupted``."""
    try:
        loop = open_job(loop_id)
    except KeyError:
        return None
    if not loop.read():
        return None
    return loop.repair_if_stale(ACTIVE, error_prefix='assist loop')


def _write_trigger(loop_id: str) -> None:
    target = jobs_root() / TRIGGER_FILE
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps({'loop_id': loop_id}), encoding='utf-8')
    tmp.replace(target)


def pending_trigger_mtime() -> float | None:
    """When the bound project's unclaimed trigger was written; ``None`` when there is none."""
    try:
        return (jobs_root() / TRIGGER_FILE).stat().st_mtime
    except FileNotFoundError:
        return None


def claim_trigger() -> str | None:
    """Take the bound project's trigger (read, then unlink: one claimant wins). The loop id
    it named, or ``None`` when nothing is pending or the file is unusable."""
    target = jobs_root() / TRIGGER_FILE
    try:
        raw = target.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    with contextlib.suppress(FileNotFoundError):
        target.unlink()
    try:
        loop_id = json.loads(raw)['loop_id']
    except (ValueError, KeyError, TypeError):
        return None
    return loop_id if isinstance(loop_id, str) else None


def _busy(live: list[str]) -> Any:
    return rejection(
        409,
        'assist_loop_busy',
        'an auto-label loop is already running for this project; cancel it or wait',
        loop_id=live[0] if live else None,
    )


def create_loop(*, state: dict[str, Any], request: dict[str, Any]) -> FileJob:
    """Claim the project's loop slot, persist the loop as queued and trigger the worker.
    ``state`` is the ``queued`` state without ``loop_id``. Raises the 409
    ``assist_loop_busy`` refusal when a loop is live."""
    root = ensure_marked_dir(jobs_root())
    with exclusive_start_lock(root / 'start.lock') as acquired:
        live = live_loop_ids()
        if not acquired or live:
            raise _busy(live)
        loop_id = f'alp_{datetime.now(UTC):%Y%m%dT%H%M%S%f}_{uuid.uuid4().hex[:6]}'
        loop = open_job(loop_id)
        loop.directory.mkdir(parents=True)
        (loop.directory / 'request.json').write_text(json.dumps(request), encoding='utf-8')
        loop.clear_signals()
        loop.write({**state, 'loop_id': loop_id})
        _write_trigger(loop_id)
    return loop


def resume_loop(loop_id: str) -> FileJob:
    """Re-queue an ``interrupted``, ``paused`` or ``cancelled`` loop and trigger the worker;
    the loop replays its log and continues where it stopped."""
    loop = _existing(loop_id)
    root = ensure_marked_dir(jobs_root())
    with exclusive_start_lock(root / 'start.lock') as acquired:
        state = read_state(loop_id) or {}
        if state.get('status') not in RESUMABLE:
            raise rejection(
                409,
                'assist_loop_not_resumable',
                f'a loop that is {state.get("status")} cannot be resumed',
                loop_id=loop_id,
            )
        others = [i for i in live_loop_ids() if i != loop_id]
        if not acquired or others:
            raise _busy(others)
        loop.clear_signals()
        with contextlib.suppress(FileNotFoundError):
            (loop.directory / PAUSE_FLAG).unlink()
        loop.update(
            status='queued',
            error=None,
            stop_reason=None,
            pause_reason=None,
            finished_at=None,
            updated_at=now_iso(),
        )
        _write_trigger(loop_id)
    return loop


def _existing(loop_id: str) -> FileJob:
    try:
        loop = open_job(loop_id)
    except KeyError:
        loop = None
    if loop is None or not loop.read():
        raise rejection(404, 'not_found', f'no auto-label loop {loop_id}')
    return loop


def cancel_loop(loop_id: str) -> bool:
    """Ask a live loop to stop. A running one finishes its requests in flight and ends
    ``cancelled``; one still waiting for the worker is cancelled at once. ``False`` when
    there is no live loop of that id."""
    try:
        loop = open_job(loop_id)
    except KeyError:
        return False
    if not loop.is_live(ACTIVE):
        return False
    loop.request_cancel()
    if loop.read().get('status') == 'queued':
        loop.update(status='cancelled', finished_at=now_iso(), updated_at=now_iso())
        trigger = jobs_root() / TRIGGER_FILE
        with contextlib.suppress(FileNotFoundError, ValueError, KeyError):
            if json.loads(trigger.read_text(encoding='utf-8'))['loop_id'] == loop_id:
                trigger.unlink()
    return True


__all__ = [
    'ACTIVE',
    'RESUMABLE',
    'cancel_loop',
    'claim_trigger',
    'create_loop',
    'jobs_root',
    'live_loop_ids',
    'loop_live',
    'open_job',
    'pending_trigger_mtime',
    'read_state',
    'reconcile_orphaned_jobs',
    'require_no_live_loop',
    'resume_loop',
]
