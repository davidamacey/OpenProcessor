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

**Per-pool single-fetch design (load-bearing):** not every scorer wants the
same items. Each scorer declares a :attr:`~item_scores.base.CropScorer.pool`
(``'residual'`` or ``'probe_scored'`` — see ``base.py`` for the semantics),
and this module builds each *needed* pool exactly ONCE per job — never
unconditionally, and never once per scorer — then hands each scorer the
pool it asked for:

* ``'residual'`` — :func:`clustering.embedding_reduce.fetch_residual_embeddings_parallel`,
  unchanged from the original single-pool design. The OpenSearch read (with
  embeddings) is the dominant cost here (350k crops x 1024-d f32 = 1.43 GB),
  not the math, so scorers sharing this pool (``uniqueness``, ``near_dup``)
  share one fetch. ``test_holdout=true`` crops are excluded via an explicit
  ``must_not`` (belt and suspenders — they're also implicitly excluded
  because ``fetch_residual_embeddings_parallel`` already drops
  ``class_validated: true`` crops, and a crop can only be
  ``test_holdout=true`` if it was ``class_validated=true`` at freeze time;
  see the reference review module's ``freeze_test_holdout``).
* ``'probe_scored'`` — ids only (no embeddings — a ``(n, 0)`` array is
  handed to the scorer), every item with ``probe_pred_class`` set, minus
  ``test_holdout`` / ``class_excluded`` / ``class_validated``. This is the
  pool ``mistakenness`` needs: it audits machine-confident labels via
  ``probe_pred_*`` scalar fields, never embeddings, so the residual pool
  (which excludes exactly the confidently-labeled items mistakenness
  exists to check) was the wrong pool for it (fixed 2026-09-25 — see
  ``git log`` for the incident: 1,351 residual items yielded
  ``n_scored=1``, ``n_skipped_outside_probe_classes=1350``, while 7,936
  items were actually probe-scored).

``state.total`` / ``state.processed`` are the SUM of each requested
scorer's own pool size (a scorer sharing a pool with another requested
scorer counts that pool's size again — this is "work units across
requested scorers", not "distinct items touched this job"). Per-scorer
detail (which pool, how many items in it, how many it actually wrote) is
in ``state.results[scorer_name]`` — additive fields on the existing shape,
not a breaking change to ``/curation/scores/status``.
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

import numpy as np

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


def _lock_file() -> Path:
    return _state_dir() / 'start.lock'


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
    state without racing the background task's own progress. That
    in-process ordering isn't enough on its own though — ``yolo-api`` runs
    under ``--workers=N``, so two calls landing on *different* worker
    processes at nearly the same instant could both pass ``_is_busy()``
    before either has written ``'running'``. The
    :func:`~src.services.curation.job_lock.exclusive_start_lock` makes the
    whole check-and-claim atomic across processes too, not just within
    one (2026-09-25 fix — this gap predates the probe job's file-backed
    rewrite that surfaced it; both now share the same fix).
    """
    from src.services.curation.job_lock import exclusive_start_lock

    with exclusive_start_lock(_lock_file()) as acquired:
        if not acquired:
            raise RuntimeError('scoring job already in progress')
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


_PROBE_SCORED_PAGE_SIZE = 2000
_PROBE_SCORED_SCROLL_TTL = '2m'


async def _fetch_probe_scored_ids(client: AsyncOpenSearch) -> list[str]:
    """The ``'probe_scored'`` pool: ids only, no embeddings.

    Every item the probe has an opinion on (``exists probe_pred_class``),
    minus ``test_holdout`` (frozen eval set), ``class_excluded`` (human
    said "not usable"), and ``class_validated`` (human-validated labels are
    human-owned — not audited by an automated mistakenness pass). Full
    scroll, no cap — a silent partial pool here would silently under-audit
    machine labels, the exact failure mode this pool exists to fix.
    """
    from src.config.curation import IndexRole, get_curation_config, index_name

    index = index_name(get_curation_config(), IndexRole.ITEMS)
    query = {
        'bool': {
            'filter': [{'exists': {'field': 'probe_pred_class'}}],
            'must_not': [
                {'term': {'test_holdout': True}},
                {'term': {'class_excluded': True}},
                {'term': {'class_validated': True}},
            ],
        },
    }
    ids: list[str] = []
    body: dict[str, Any] = {
        'size': _PROBE_SCORED_PAGE_SIZE,
        'query': query,
        '_source': False,
    }
    resp = await client.search(index=index, body=body, scroll=_PROBE_SCORED_SCROLL_TTL)
    scroll_id = resp.get('_scroll_id')
    try:
        while True:
            hits = resp['hits']['hits']
            if not hits:
                break
            ids.extend(h['_id'] for h in hits)
            if not scroll_id:
                break
            resp = await client.scroll(scroll_id=scroll_id, scroll=_PROBE_SCORED_SCROLL_TTL)
            scroll_id = resp.get('_scroll_id')
    finally:
        if scroll_id:
            with contextlib.suppress(Exception):
                await client.clear_scroll(scroll_id=scroll_id)
    return ids


async def _build_pools(
    opensearch: AsyncOpenSearch,
    needed: set[str],
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Fetch each needed pool exactly once. ``needed`` is the set of
    ``pool`` values actually declared by the requested scorers — a pool
    nothing asked for is never fetched."""
    from src.services.curation.clustering.embedding_reduce import fetch_residual_embeddings_parallel

    pools: dict[str, tuple[list[str], np.ndarray]] = {}
    if 'residual' in needed:
        pools['residual'] = await fetch_residual_embeddings_parallel(
            opensearch,
            # Belt-and-suspenders test_holdout exclusion — see module
            # docstring for why this is already implied.
            extra_must=[{'bool': {'must_not': [{'term': {'test_holdout': True}}]}}],
        )
    if 'probe_scored' in needed:
        probe_ids = await _fetch_probe_scored_ids(opensearch)
        pools['probe_scored'] = (probe_ids, np.zeros((len(probe_ids), 0), dtype=np.float32))
    return pools


async def run_scoring_job(
    job_id: str,
    opensearch: AsyncOpenSearch,
    scorer_names: list[str],
) -> None:
    """Actual compute body: fetch each pool the requested scorers need
    (once each), then one scorer at a time, writing results back via bulk
    update + checkpointing state.json between scorers so
    ``/curation/scores/status`` reflects real progress.

    A dedicated (non-underscore) symbol so tests can monkeypatch it
    wholesale for deterministic job-lifecycle testing without racing a
    real background task against synchronous test-client HTTP calls.
    """
    from src.services.curation.item_scores import get_scorer

    state = _read_state()
    if state.job_id != job_id:
        # Superseded by a newer job (shouldn't happen — start_job is a
        # singleton gate) — bail out rather than clobber someone else's run.
        return
    _touch_heartbeat()
    try:
        scorers = {name: get_scorer(name) for name in scorer_names}
        needed_pools = {scorer.pool for scorer in scorers.values()}
        pools = await _build_pools(opensearch, needed_pools)

        state.total = sum(len(pools[scorer.pool][0]) for scorer in scorers.values())
        _atomic_write(state)
        _touch_heartbeat()

        results: dict[str, Any] = {}
        for name in scorer_names:
            if is_cancelled():
                state.status = 'cancelled'
                state.finished_at = time.time()
                _atomic_write(state)
                return
            scorer = scorers[name]
            ids, embeddings = pools[scorer.pool]
            result = await scorer.score(ids, embeddings, opensearch=opensearch)
            await bulk_write_result(opensearch, result)
            results[name] = {
                'n_scored': result.n_scored,
                'extra': result.extra,
                'pool': scorer.pool,
                'pool_size': len(ids),
            }
            state.results = results
            state.processed += len(ids)
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
