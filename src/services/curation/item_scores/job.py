"""Singleton job runner for ``POST /curation/scores/compute``.

Mirrors :mod:`src.services.curation.auto_label_job`'s state.json / heartbeat /
cancel.flag file-backed conventions, simplified for in-process execution —
there is no separate scores worker container; the job runs as an
``asyncio`` background task inside the yolo-api process itself, started
by the router handler and polled via the same state-file pattern the
labeler already knows how to render.

Directory resolved lazily via ``OP_SCORES_STATE_DIR`` (default ``/jobs/scores``)
so tests can override with ``monkeypatch.setenv`` + ``tmp_path`` without
reimporting — same convention as ``train_jobs._resolve_jobs_dir``.

**Single-fetch design (load-bearing):** embeddings are fetched
ONCE via :func:`clustering.embedding_reduce.fetch_residual_embeddings_parallel`
and the same matrix is handed to every enabled scorer — the OpenSearch read
is the dominant cost (350k crops x 1024-d f32 = 1.43 GB), not the math.
``test_holdout=true`` crops are excluded via an explicit ``must_not`` (belt
and suspenders — they're also implicitly excluded because
``fetch_residual_embeddings_parallel`` already drops ``class_validated:
true`` crops, and a crop can only be ``test_holdout=true`` if it was
``class_validated=true`` at freeze time; see the reference review module's ``freeze_test_holdout``).
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

from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.item_scores.base import ScoreResult


logger = get_logger(__name__)


_HEARTBEAT_STALE_S = 30.0

# Reference to the scheduled background task — asyncio only holds a weak
# reference internally, so an unreferenced task can be garbage-collected
# mid-run. Module-level singleton matches the one-job-at-a-time contract
# start_job()/_is_busy() already enforce via the state file.
_active_task: asyncio.Task[None] | None = None


def _state_dir() -> Path:
    """Resolved fresh each call so tests can override via monkeypatch."""
    return Path(os.environ.get('OP_SCORES_STATE_DIR', '/jobs/scores'))


def _state_file() -> Path:
    return _state_dir() / 'state.json'


def _cancel_flag() -> Path:
    return _state_dir() / 'cancel.flag'


def _heartbeat_file() -> Path:
    return _state_dir() / 'heartbeat'


@dataclass
class _JobState:
    job_id: str = ''
    status: str = 'idle'  # 'idle' | 'running' | 'completed' | 'failed' | 'cancelled'
    scorers: list[str] = field(default_factory=list)
    processed: int = 0
    total: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str | None = None
    results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ensure_dir() -> None:
    _state_dir().mkdir(parents=True, exist_ok=True)


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
    # No heartbeat yet just means the task hasn't ticked once — still busy.
    return age is None or age <= _HEARTBEAT_STALE_S


def reconcile_orphaned_jobs() -> bool:
    """Startup-only repair: see :mod:`src.services.curation.job_reconcile`.

    Called from ``src.main``'s lifespan before any request is served, so
    nothing in this process can legitimately hold ``status='running'``
    yet — a leftover 'running' state.json is necessarily orphaned by a
    prior process. Returns True if the file was rewritten.
    """
    from src.services.curation.job_reconcile import reconcile_stale_running

    return reconcile_stale_running(
        _state_file(),
        _heartbeat_file(),
        stale_s=_HEARTBEAT_STALE_S,
        error_prefix='scoring job',
    )


def get_state() -> dict[str, Any]:
    """Read-only snapshot, with stale-heartbeat repair (mirrors
    ``auto_label_job.get_state``'s liveness contract)."""
    state = _read_state()
    if state.status == 'running':
        age = _heartbeat_age()
        if age is not None and age > _HEARTBEAT_STALE_S:
            state.status = 'failed'
            state.error = state.error or f'scoring job heartbeat stale ({age:.1f}s ago)'
            state.finished_at = state.finished_at or time.time()
            _atomic_write(state)
    return state.to_dict()


def start_job(opensearch: AsyncOpenSearch, scorer_names: list[str]) -> dict[str, Any]:
    """Write 'running' state synchronously, then schedule the background
    task. Raises ``RuntimeError`` if a run is already in flight — the
    router surfaces this as HTTP 409.

    Deliberately NOT ``async``: the busy-check + state transition to
    'running' happens before any ``await`` point, so a second call issued
    immediately after (even from a concurrent request) sees the 'running'
    state without racing the background task's own progress.
    """
    if _is_busy():
        raise RuntimeError('scoring job already in progress')
    _ensure_dir()
    with contextlib.suppress(FileNotFoundError):
        _cancel_flag().unlink()
    with contextlib.suppress(FileNotFoundError):
        _heartbeat_file().unlink()

    job_id = uuid.uuid4().hex
    state = _JobState(
        job_id=job_id,
        status='running',
        scorers=list(scorer_names),
        started_at=time.time(),
    )
    _atomic_write(state)
    global _active_task  # noqa: PLW0603 - singleton task handle, mirrors auto_label_job's module globals
    _active_task = asyncio.create_task(run_scoring_job(job_id, opensearch, scorer_names))
    return state.to_dict()


def cancel_job() -> bool:
    """Touch the cancel flag. Returns True if a run was active."""
    if not _is_busy():
        return False
    _ensure_dir()
    _cancel_flag().touch()
    return True


async def bulk_write_result(
    opensearch: AsyncOpenSearch,
    result: ScoreResult,
    *,
    chunk_size: int = 500,
) -> None:
    from src.config.curation import IndexRole, get_curation_config, index_name

    index = index_name(get_curation_config(), IndexRole.ITEMS)
    items = list(result.fields.items())
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        bulk: list[dict[str, Any]] = []
        for crop_id, doc in chunk:
            bulk.append({'update': {'_index': index, '_id': crop_id}})
            bulk.append({'doc': doc})
        if bulk:
            await opensearch.bulk(body=bulk, refresh=False)


async def run_scoring_job(
    job_id: str,
    opensearch: AsyncOpenSearch,
    scorer_names: list[str],
) -> None:
    """Actual compute body: single embedding fetch, then one scorer at a
    time, writing results back via bulk update + checkpointing state.json
    between scorers so ``/curation/scores/status`` reflects real progress.

    A dedicated (non-underscore) symbol so tests can monkeypatch it
    wholesale for deterministic job-lifecycle testing without racing a
    real background task against synchronous test-client HTTP calls.
    """
    from src.services.curation.clustering.embedding_reduce import fetch_residual_embeddings_parallel
    from src.services.curation.item_scores import get_scorer

    state = _read_state()
    if state.job_id != job_id:
        # Superseded by a newer job (shouldn't happen — start_job is a
        # singleton gate) — bail out rather than clobber someone else's run.
        return
    _touch_heartbeat()
    try:
        ids, embeddings = await fetch_residual_embeddings_parallel(
            opensearch,
            # Belt-and-suspenders test_holdout exclusion — see module
            # docstring for why this is already implied.
            extra_must=[{'bool': {'must_not': [{'term': {'test_holdout': True}}]}}],
        )
        state.total = len(ids)
        _atomic_write(state)
        _touch_heartbeat()

        results: dict[str, Any] = {}
        for name in scorer_names:
            if is_cancelled():
                state.status = 'cancelled'
                state.finished_at = time.time()
                _atomic_write(state)
                return
            scorer = get_scorer(name)
            result = await scorer.score(ids, embeddings, opensearch=opensearch)
            await bulk_write_result(opensearch, result)
            results[name] = {'n_scored': result.n_scored, 'extra': result.extra}
            state.results = results
            state.processed = state.total
            _atomic_write(state)
            _touch_heartbeat()

        state.status = 'completed'
        state.finished_at = time.time()
        _atomic_write(state)
    except asyncio.CancelledError:
        state.status = 'cancelled'
        state.finished_at = time.time()
        _atomic_write(state)
        raise
    except Exception as exc:
        logger.error('curation_scores_job_failed', job_id=job_id, error=str(exc))
        state.status = 'failed'
        state.error = str(exc)
        state.finished_at = time.time()
        _atomic_write(state)


async def compute_coverage(opensearch: AsyncOpenSearch) -> dict[str, Any]:
    """Per-scorer ``{n_scored, total, pct}`` for ``GET /curation/scores/coverage``."""
    from src.config.curation import IndexRole, get_curation_config, index_name
    from src.services.curation.item_scores import SCORER_METADATA

    index = index_name(get_curation_config(), IndexRole.ITEMS)
    total_resp = await opensearch.count(index=index, body={'query': {'match_all': {}}})
    total = int(total_resp.get('count', 0))

    coverage: dict[str, Any] = {}
    for scorer_id, meta in SCORER_METADATA.items():
        field_name = meta['requires_field']
        try:
            resp = await opensearch.count(
                index=index, body={'query': {'exists': {'field': field_name}}}
            )
            n = int(resp.get('count', 0))
        except Exception as exc:
            logger.warning(
                'curation_scores_coverage_count_failed', scorer=scorer_id, error=str(exc)
            )
            n = 0
        coverage[scorer_id] = {
            'field': field_name,
            'n_scored': n,
            'total': total,
            'pct': round(100.0 * n / total, 2) if total else 0.0,
        }
    return coverage


__all__ = [
    'bulk_write_result',
    'cancel_job',
    'compute_coverage',
    'get_state',
    'is_cancelled',
    'reconcile_orphaned_jobs',
    'run_scoring_job',
    'start_job',
]
