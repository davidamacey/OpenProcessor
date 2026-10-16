"""The file-backed assist run: one run per project at a time, state on the shared
``/jobs`` volume, executed as a task in the API process.

State lives under ``<OP_ASSIST_JOBS_DIR>/projects/<slug>/<run_id>/``
(:class:`~src.services.curation.file_job.FileJob`): ``state.json``, ``request.json``,
and ``members.json`` (the resolved scope ids, for the staleness check). Every API
worker process sees the same files, so status and cancel work whichever process a
request lands on. A restart ends a live run as ``interrupted``
(:func:`reconcile_orphaned_jobs`); the user re-runs it. The newest
:data:`KEEP_RUNS` runs are kept.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.env import env_path
from src.config.project_context import ensure_marked_dir, mark_project_dir, project_jobs_dir
from src.core.logging import get_logger
from src.services.curation.assist.models import ACTIVE_STATUSES, AssistRunState
from src.services.curation.file_job import FileJob
from src.services.curation.job_lock import exclusive_start_lock
from src.services.rejections import rejection


if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

ACTIVE = ACTIVE_STATUSES
KEEP_RUNS = 100


def jobs_root() -> Path:
    """The bound project's assist runs dir (resolved per call so tests can
    ``monkeypatch.setenv``)."""
    return mark_project_dir(project_jobs_dir(env_path('OP_ASSIST_JOBS_DIR', '/jobs/assist')))


def _job(run_id: str) -> FileJob:
    # run ids are minted here; never let a path segment through.
    if not run_id or '/' in run_id or '\\' in run_id or run_id.startswith('.'):
        raise KeyError(run_id)
    return FileJob(jobs_root() / run_id)


def _run_dirs() -> list[Path]:
    root = jobs_root()
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def live_run_ids() -> list[str]:
    return [p.name for p in _run_dirs() if FileJob(p).is_live(ACTIVE)]


def reconcile_orphaned_jobs() -> bool:
    """Startup repair for the bound project: a run left active by a dead process
    becomes ``interrupted``."""
    repaired = False
    for path in _run_dirs():
        repaired |= FileJob(path).reconcile(active_statuses=ACTIVE, error_prefix='assist run')
    return repaired


def create_run(
    *,
    state: dict[str, Any],
    request: dict[str, Any],
    members: list[str] | None,
) -> FileJob:
    """Claim the per-project singleton and persist the run. ``state`` is the
    ``queued`` :class:`AssistRunState` dump without ``run_id``. Raises the 409
    ``assist_busy`` refusal when a run is live."""
    root = ensure_marked_dir(jobs_root())
    with exclusive_start_lock(root / 'start.lock') as acquired:
        live = live_run_ids()
        if not acquired or live:
            raise rejection(
                409,
                'assist_busy',
                'an assist run is already running for this project; cancel it or wait',
                run_id=live[0] if live else None,
            )
        run_id = f'asr_{datetime.now(UTC):%Y%m%dT%H%M%S%f}_{uuid.uuid4().hex[:6]}'
        job = _job(run_id)
        job.directory.mkdir(parents=True)
        (job.directory / 'request.json').write_text(json.dumps(request), encoding='utf-8')
        if members is not None:
            (job.directory / 'members.json').write_text(json.dumps(members), encoding='utf-8')
        job.clear_signals()
        job.write({**state, 'run_id': run_id})
        job.touch_heartbeat()
    _prune(keep=run_id)
    return job


def _prune(*, keep: str) -> None:
    dirs = _run_dirs()
    for path in dirs[: max(0, len(dirs) - KEEP_RUNS)]:
        if path.name != keep and not FileJob(path).is_live(ACTIVE):
            shutil.rmtree(path, ignore_errors=True)


def read_state(run_id: str) -> AssistRunState | None:
    try:
        job = _job(run_id)
    except KeyError:
        return None
    if not job.read():
        return None
    state = job.repair_if_stale(ACTIVE, error_prefix='assist run')
    return AssistRunState.model_validate(state)


def read_members(run_id: str) -> list[str] | None:
    """The member ids the run resolved, or ``None`` when it kept none (a filter target)."""
    try:
        path = _job(run_id).directory / 'members.json'
        data = json.loads(path.read_text(encoding='utf-8'))
    except (KeyError, OSError, ValueError):
        return None
    return [str(i) for i in data] if isinstance(data, list) else None


def list_runs(
    *, cluster_id: int | None, cluster_subid: str | None, limit: int
) -> list[AssistRunState]:
    """The newest runs, newest first, whose target or context names that scope."""
    out: list[AssistRunState] = []
    for path in reversed(_run_dirs()):
        state = read_state(path.name)
        if state is None or not _names_scope(state, cluster_id, cluster_subid):
            continue
        out.append(state)
        if len(out) >= limit:
            break
    return out


def _names_scope(state: AssistRunState, cluster_id: int | None, subid: str | None) -> bool:
    target = state.request.target
    scope = getattr(target, 'context', None) or target
    got_id = getattr(scope, 'cluster_id', None)
    got_subid = getattr(scope, 'cluster_subid', None)
    if cluster_id is None:
        return True
    return got_id == cluster_id and (subid is None or got_subid == subid)


def cancel_run(run_id: str) -> bool:
    """Touch the cancel flag of a live run; ``False`` when there is none."""
    try:
        job = _job(run_id)
    except KeyError:
        return False
    if not job.is_live(ACTIVE):
        return False
    job.request_cancel()
    return True


__all__ = [
    'ACTIVE',
    'KEEP_RUNS',
    'cancel_run',
    'create_run',
    'jobs_root',
    'list_runs',
    'live_run_ids',
    'read_members',
    'read_state',
    'reconcile_orphaned_jobs',
]
