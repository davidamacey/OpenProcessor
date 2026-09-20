"""Singleton job runner for pool-scale ``POST /curation/select/diverse`` requests
(curation-strategy plan §3.4/§7 Phase 4).

Mirrors :mod:`src.services.curation.item_scores.job`'s state.json / heartbeat
/ cancel.flag file-backed conventions (itself mirroring
``auto_label_job.py``), simplified for a single selection run instead of a
list of scorers. See ``kb_select.py``'s module docstring for *when* this
job path is used instead of answering inline — short version: the plan's
own compute budget says k-center-greedy at pool scale (n≈128k, k≈1000) is
~1-2 min CPU, too slow to block an HTTP request, so anything above a
documented sync-ops budget runs here as a backgrounded ``asyncio`` task
instead (no separate worker container, same as ``crop_scores.job``).

Directory resolved lazily via ``OP_SELECT_JOBS_DIR`` (default
``/jobs/select``) so tests can override with ``monkeypatch.setenv`` +
``tmp_path`` without reimporting.

Read-only with respect to crop documents: this job only ever *reads*
embeddings and writes its own job-state file under ``OP_SELECT_JOBS_DIR``
— it never issues an OpenSearch ``update``/``bulk`` write (plan §8
non-goal #3 / the "never writes a crop field" hard constraint for the
whole selection overlay).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np
    from opensearchpy import AsyncOpenSearch

from src.core.logging import get_logger


logger = get_logger(__name__)

_HEARTBEAT_STALE_S = 30.0

# See crop_scores.job's identical module-level comment: asyncio only holds
# a weak reference internally, so an unreferenced task can be
# garbage-collected mid-run. This keeps one alive per the singleton
# contract start_job()/_is_busy() enforce via the state file.
_active_task: asyncio.Task[None] | None = None


def _jobs_dir() -> Path:
    return Path(os.environ.get('OP_SELECT_JOBS_DIR', '/jobs/select'))


def _state_file() -> Path:
    return _jobs_dir() / 'state.json'


def _cancel_flag() -> Path:
    return _jobs_dir() / 'cancel.flag'


def _heartbeat_file() -> Path:
    return _jobs_dir() / 'heartbeat'


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


def _ensure_dir() -> None:
    _jobs_dir().mkdir(parents=True, exist_ok=True)


def _atomic_write(state: _JobState) -> None:
    _ensure_dir()
    tmp = _state_file().with_suffix('.tmp')
    tmp.write_text(json.dumps(state.to_dict(), default=str))
    tmp.replace(_state_file())


def _read_state() -> _JobState:
    try:
        raw = json.loads(_state_file().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return _JobState()
    state = _JobState()
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


def is_cancelled() -> bool:
    return _cancel_flag().exists()


def _is_busy() -> bool:
    state = _read_state()
    if state.status != 'running':
        return False
    age = _heartbeat_age()
    return age is None or age <= _HEARTBEAT_STALE_S


def get_state() -> dict[str, Any]:
    """Read-only snapshot, with stale-heartbeat repair (mirrors
    ``crop_scores.job.get_state``'s liveness contract)."""
    state = _read_state()
    if state.status == 'running':
        age = _heartbeat_age()
        if age is not None and age > _HEARTBEAT_STALE_S:
            state.status = 'failed'
            state.error = state.error or f'selection job heartbeat stale ({age:.1f}s ago)'
            state.finished_at = state.finished_at or time.time()
            _atomic_write(state)
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
    _ensure_dir()
    with contextlib.suppress(FileNotFoundError):
        _cancel_flag().unlink()
    with contextlib.suppress(FileNotFoundError):
        _heartbeat_file().unlink()

    job_id = uuid.uuid4().hex
    state = _JobState(job_id=job_id, status='running', k=k, scope=scope, started_at=time.time())
    _atomic_write(state)
    global _active_task  # noqa: PLW0603 - singleton task handle, mirrors crop_scores.job
    _active_task = asyncio.create_task(
        run_selection_job(job_id, opensearch, index, query, k, seed_crop_id, max_n)
    )
    return state.to_dict()


def cancel_job() -> bool:
    if not _is_busy():
        return False
    _ensure_dir()
    _cancel_flag().touch()
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
    _touch_heartbeat()
    try:
        if is_cancelled():
            state.status = 'cancelled'
            state.finished_at = time.time()
            _atomic_write(state)
            return

        ids, embeddings, _truncated = await fetch_pool_embeddings(
            opensearch, index, query, cap=max_n
        )
        state.n_pool = len(ids)
        _atomic_write(state)
        _touch_heartbeat()

        if not ids:
            state.status = 'completed'
            state.result = {
                'crop_ids': [],
                'method': 'kcenter_greedy',
                'version': 'v1',
                'n_pool': 0,
            }
            state.finished_at = time.time()
            _atomic_write(state)
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
        _atomic_write(state)
    except asyncio.CancelledError:
        state.status = 'cancelled'
        state.finished_at = time.time()
        _atomic_write(state)
        raise
    except Exception as exc:
        logger.error('kb_select_job_failed', job_id=job_id, error=str(exc))
        state.status = 'failed'
        state.error = str(exc)
        state.finished_at = time.time()
        _atomic_write(state)


__all__ = [
    'cancel_job',
    'get_state',
    'is_cancelled',
    'run_selection_job',
    'start_job',
]
