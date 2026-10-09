"""Singleton job runner for pool-scale ``POST /curation/select/diverse`` requests.

Mirrors :mod:`src.services.curation.item_scores.job`'s state.json / heartbeat
/ cancel.flag file-backed conventions (itself mirroring
``auto_label_job.py``), simplified for a single selection run instead of a
list of scorers. See the select router's module docstring for *when* this
job path is used instead of answering inline — short version: the
compute budget says k-center-greedy at pool scale (n≈128k, k≈1000) is
~1-2 min CPU, too slow to block an HTTP request, so anything above a
documented sync-ops budget runs here as a backgrounded ``asyncio`` task
instead (no separate worker container, same as ``crop_scores.job``).

Directory resolved lazily via ``OP_SELECT_JOBS_DIR`` (default
``/jobs/select``) so tests can override with ``monkeypatch.setenv`` +
``tmp_path`` without reimporting.

Read-only with respect to crop documents: this job only ever *reads*
embeddings and writes its own job-state file under ``OP_SELECT_JOBS_DIR``
— it never issues an OpenSearch ``update``/``bulk`` write (the "never
writes a crop field" hard constraint for the whole selection overlay).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np
    from opensearchpy import AsyncOpenSearch

from pathlib import Path

from src.config.curation import get_curation_config
from src.config.project_context import project_jobs_dir
from src.core.logging import get_logger
from src.services.curation.file_job import HEARTBEAT_STALE_S, FileJob


logger = get_logger(__name__)

_ACTIVE = frozenset({'running'})

# See crop_scores.job's identical module-level comment: asyncio only holds
# a weak reference internally, so an unreferenced task can be
# garbage-collected mid-run. This keeps one alive per the singleton
# contract start_job()/_is_busy() enforce via the state file.
_active_tasks: dict[str, asyncio.Task[None]] = {}  # per project slug


def _jobs_dir() -> Path:
    return project_jobs_dir(Path(os.environ.get('OP_SELECT_JOBS_DIR', '/jobs/select')))


def _job() -> FileJob:
    return FileJob(_jobs_dir())


@dataclass
class _JobState:
    job_id: str = ''
    status: str = 'idle'  # 'idle' | 'running' | 'completed' | 'failed' | 'cancelled'
    k: int = 0
    scope: dict[str, Any] = field(default_factory=dict)
    n_pool: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_state() -> _JobState:
    state = _JobState()
    for k, v in _job().read().items():
        if hasattr(state, k):
            setattr(state, k, v)
    return state


def is_cancelled() -> bool:
    return _job().cancel_requested()


def _is_busy() -> bool:
    return _job().is_live(_ACTIVE)


def reconcile_orphaned_jobs() -> bool:
    """Startup-only repair: see :mod:`src.services.curation.job_reconcile`.

    Called from ``src.main``'s lifespan before any request is served —
    nothing in this process can legitimately hold ``status='running'``
    yet, so a leftover 'running' state.json is necessarily orphaned by a
    prior process. Returns True if the file was rewritten.
    """
    return _job().reconcile(active_statuses=_ACTIVE, error_prefix='selection job')


def get_state() -> dict[str, Any]:
    """Read-only snapshot, with stale-heartbeat repair (mirrors
    ``crop_scores.job.get_state``'s liveness contract)."""
    state = _read_state()
    if state.status == 'running':
        age = _job().heartbeat_age()
        if age is not None and age > HEARTBEAT_STALE_S:
            state.status = 'failed'
            state.error = state.error or f'selection job heartbeat stale ({age:.1f}s ago)'
            state.finished_at = state.finished_at or time.time()
            _job().write(state.to_dict())
    return state.to_dict()


def start_job(
    opensearch: AsyncOpenSearch,
    *,
    index: str,
    query: dict[str, Any],
    k: int,
    seed_crop_id: str | None,
    scope: dict[str, Any],
    max_n: int,
) -> dict[str, Any]:
    """Write 'running' state synchronously, then schedule the background
    task. Raises ``RuntimeError`` if a run is already in flight — the
    router surfaces this as HTTP 409 (same contract as
    ``crop_scores.job.start_job``).
    """
    if _is_busy():
        raise RuntimeError('selection job already in progress')
    _job().clear_signals()

    job_id = uuid.uuid4().hex
    state = _JobState(job_id=job_id, status='running', k=k, scope=scope, started_at=time.time())
    _job().write(state.to_dict())
    _active_tasks[get_curation_config().project_slug] = asyncio.create_task(
        run_selection_job(job_id, opensearch, index, query, k, seed_crop_id, max_n)
    )
    return state.to_dict()


def cancel_job() -> bool:
    if not _is_busy():
        return False
    _job().request_cancel()
    return True


async def run_selection_job(
    job_id: str,
    opensearch: AsyncOpenSearch,
    index: str,
    query: dict[str, Any],
    k: int,
    seed_crop_id: str | None,
    max_n: int,
) -> None:
    """Actual compute body: one pool fetch (uncapped by the sync-path
    ``OP_SELECT_MAX_N``, only by the generous job-path ``max_n`` ceiling),
    one ``k_center_greedy`` call, done. A dedicated (non-underscore) symbol
    so tests can monkeypatch it wholesale for deterministic job-lifecycle
    testing without racing a real background task against synchronous
    TestClient calls — same reasoning as ``crop_scores.job.run_scoring_job``.

    Cancellation is checked once before the (uninterruptible) greedy call
    starts, not mid-computation — ``k_center_greedy``'s tight numpy loop
    has no natural yield point to check a flag inside, and
    ``crop_scores.job`` has the same granularity (checks between scorers,
    not mid-scorer). A job already past this point runs to completion.
    """
    from src.services.curation.selection import k_center_greedy
    from src.services.curation.selection.pool_fetch import fetch_pool_embeddings, l2_normalize

    state = _read_state()
    if state.job_id != job_id:
        return
    _job().touch_heartbeat()
    try:
        if is_cancelled():
            state.status = 'cancelled'
            state.finished_at = time.time()
            _job().write(state.to_dict())
            return

        ids, embeddings, _truncated = await fetch_pool_embeddings(
            opensearch, index, query, cap=max_n
        )
        state.n_pool = len(ids)
        _job().write(state.to_dict())
        _job().touch_heartbeat()

        if not ids:
            state.status = 'completed'
            state.result = {
                'crop_ids': [],
                'method': 'kcenter_greedy',
                'version': 'v1',
                'n_pool': 0,
            }
            state.finished_at = time.time()
            _job().write(state.to_dict())
            return

        seed_idx: int | None = None
        if seed_crop_id is not None:
            with contextlib.suppress(ValueError):
                seed_idx = ids.index(seed_crop_id)

        selected_idx: np.ndarray = k_center_greedy(l2_normalize(embeddings), k, seed_idx=seed_idx)
        crop_ids = [ids[i] for i in selected_idx.tolist()]

        state.status = 'completed'
        state.result = {
            'crop_ids': crop_ids,
            'method': 'kcenter_greedy',
            'version': 'v1',
            'n_pool': len(ids),
        }
        state.finished_at = time.time()
        _job().write(state.to_dict())
    except asyncio.CancelledError:
        state.status = 'cancelled'
        state.finished_at = time.time()
        _job().write(state.to_dict())
        raise
    except Exception as exc:
        logger.error('curation_select_job_failed', job_id=job_id, error=str(exc))
        state.status = 'failed'
        state.error = str(exc)
        state.finished_at = time.time()
        _job().write(state.to_dict())


__all__ = [
    'cancel_job',
    'get_state',
    'is_cancelled',
    'reconcile_orphaned_jobs',
    'run_selection_job',
    'start_job',
]
