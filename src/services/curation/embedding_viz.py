"""2-D UMAP projection for **visualization only** (curation-strategy plan
§2.7/§3.5/§7 Phase 5).

Non-negotiable design rules (plan §2.7 / §8 non-goal #4):

1. **Own persisted state slot.** Never touches
   ``embedding_reduce.UMAP_STATE_JOBLIB_PATH{,_CUML}`` or the
   ``op_umap_state`` index — those belong to the *retired* clustering
   reducer (``docs/design/clustering_methods.md`` §2.1: UMAP collapsed
   most of the residual pool into one mega-cluster and was retired for
   clustering). This module's state lives at ``umap_viz_state.joblib``
   (disk) and the ``op_umap_viz_state`` index — deliberately distinct
   names, never imported by/from ``embedding_reduce.py`` /
   ``curation_umap.py`` / ``clustering/orchestrator.py`` /
   ``clustering/methods/``.
2. **Never fits on a request path.** ``fit_projection`` (the only
   function importing ``umap`` / calling ``.fit_transform``) is reachable
   solely from :func:`run_projection_job`, scheduled only by
   :func:`start_job` (``POST /curation/viz/projection/rebuild``).
   ``GET /curation/viz/projection`` calls only
   :func:`get_cached_projection`, a plain OpenSearch ``search`` over
   already-written ``viz_x``/``viz_y`` fields — enforced by
   ``tests/curation/test_embedding_viz.py::test_get_cached_projection_never_triggers_a_fit``.
3. **Color comes from the real ``cluster_id`` field, computed by FAISS
   IVF elsewhere** — this module doesn't invent a second cluster concept
   and never assigns or writes ``cluster_id``; it only reads it verbatim
   off the crop doc for the consumer to color by.
4. **Only writes ``viz_x`` / ``viz_y`` / ``viz_projection_version``**
   (the three fields declared by
   :func:`src.clients.curation_opensearch.ensure_items_viz_fields`).
   Never ``cluster_id`` / ``cluster_subid`` / ``cluster_distance`` (plan §8
   non-goal #3, guarded by
   ``tests/curation/test_embedding_viz.py::test_writes_never_include_cluster_fields``).

Embedding fetch reuses the two existing helpers rather than a third
scroll/PIT-fetch implementation (plan §3.5 note): ``scope='residual'``
delegates to
:func:`src.services.curation.clustering.embedding_reduce.fetch_residual_v6_embeddings_parallel`
(same pool + ``CONFIDENT_CLASS_SOURCES``/``class_excluded`` gate
``item_scores.job`` uses); ``scope='cluster'`` delegates to
:func:`src.services.curation.selection.pool_fetch.fetch_pool_embeddings`
with a ``cluster_id`` term filter (same building block the select router
uses). Both paths already exclude ``test_holdout=true`` crops.

Job-runner pattern mirrors :mod:`src.services.curation.item_scores.job`'s
singleton state.json/heartbeat/cancel.flag conventions rather than
:mod:`src.services.curation.selection.job`'s, since (like
``item_scores.job``, unlike the read-only ``selection.job``) this job
*writes* fields back via a bulk update. Every rebuild does one full fresh
``fit_transform`` and writes coordinates; persisting the fitted reducer
is for provenance only, not a functional requirement.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import get_curation_config
from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

ITEMS_INDEX = get_curation_config().items_index

# Own OpenSearch index for run metadata -- deliberately NOT the retired
# clustering reducer's index (embedding_reduce.py). Routed through
# CurationConfig like every other index name so a deployment renaming
# its indexes via env vars doesn't leave this one behind.
UMAP_VIZ_STATE_INDEX = get_curation_config().umap_viz_state_index

# State dir shared with the rest of the curation worker fleet (same
# CurationConfig.state_dir embedding_reduce.py reads), but a distinct
# filename -- never umap_state.joblib / umap_state_cuml.joblib.
VIZ_STATE_JOBLIB_PATH = str(Path(get_curation_config().state_dir) / 'umap_viz_state.joblib')

# Viz-only UMAP hyperparameters. min_dist is higher than the clustering
# reducer's 0.0 (UMAP_MIN_DIST in embedding_reduce.py) -- a viz scatter
# benefits from a bit more visual spread between points; this has zero
# effect on clustering since nothing here ever feeds cluster_id.
VIZ_N_COMPONENTS = 2
VIZ_N_NEIGHBORS = 15
VIZ_MIN_DIST = 0.1
VIZ_METRIC = 'cosine'
VIZ_RANDOM_STATE = 42

# Bumped whenever the fit recipe (hyperparameters/embedding field) changes
# so a served point's provenance is unambiguous -- mirrors item_scores'
# per-scorer ``version`` stamp.
VIZ_PROJECTION_VERSION = 'umap_viz_v1'

EMBEDDING_FIELD = 'pe_embedding'

# max_result_window is 10000; get_cached_projection() pages with search_after
# in chunks of this size instead of a single oversized `size: max_points` (F-2).
_VIZ_PROJECTION_PAGE_SIZE = 5000

# Job pool cap. Residual-scope fetch has no built-in cap (unlike
# selection.pool_fetch's `cap` kwarg), so this module samples down to
# max_n *after* the fetch, deterministically (seeded) to avoid shard bias.
DEFAULT_MAX_N = 20_000

_HEARTBEAT_STALE_S = 30.0

# See item_scores.job's identical comment: asyncio only holds a weak
# reference internally, so an unreferenced task can be garbage-collected
# mid-run. Kept alive per the singleton contract start_job()/_is_busy()
# enforce via the state file.
_active_task: asyncio.Task[None] | None = None


def _jobs_dir() -> Path:
    """Resolved fresh each call so tests can override via monkeypatch
    (same convention as ``item_scores.job._state_dir``)."""
    return Path(os.environ.get('OP_VIZ_JOBS_DIR', '/jobs/viz'))


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
    scope: str = ''
    cluster_id: int | None = None
    n_pool: int = 0
    n_written: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str | None = None
    projection_version: str | None = None

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


_HEARTBEAT_TICK_S = 10.0


async def _heartbeat_ticker() -> None:
    """Keep the heartbeat fresh across the whole job, not just between
    phases. The pool fetch (residual scope: no page cap, real measured
    pace is single-digit-seconds per ~250-crop page — minutes at
    DEFAULT_MAX_N) and the UMAP fit itself (module docstring: up to
    ~10-30 min CPU at 128k) can each legitimately run past
    _HEARTBEAT_STALE_S on their own. Without a ticker, get_state()'s
    stale-heartbeat repair marks a genuinely-running job 'failed'
    mid-phase, and _is_busy() (same staleness check) would then let a
    second job start concurrently against the same state file."""
    while True:
        await asyncio.sleep(_HEARTBEAT_TICK_S)
        _touch_heartbeat()


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


def reconcile_orphaned_jobs() -> bool:
    """Startup-only repair: see :mod:`src.services.curation.job_reconcile`.

    Called from ``src.main``'s lifespan before any request is served —
    nothing in this process can legitimately hold ``status='running'``
    yet, so a leftover 'running' state.json is necessarily orphaned by a
    prior process. Returns True if the file was rewritten.
    """
    from src.services.curation.job_reconcile import reconcile_stale_running

    return reconcile_stale_running(
        _state_file(),
        _heartbeat_file(),
        stale_s=_HEARTBEAT_STALE_S,
        error_prefix='viz projection job',
    )


def get_state() -> dict[str, Any]:
    """Read-only snapshot, with stale-heartbeat repair (mirrors
    ``item_scores.job.get_state``'s liveness contract)."""
    state = _read_state()
    if state.status == 'running':
        age = _heartbeat_age()
        if age is not None and age > _HEARTBEAT_STALE_S:
            state.status = 'failed'
            state.error = state.error or f'viz projection job heartbeat stale ({age:.1f}s ago)'
            state.finished_at = state.finished_at or time.time()
            _atomic_write(state)
    return state.to_dict()


def start_job(
    opensearch: AsyncOpenSearch,
    *,
    scope: str,
    cluster_id: int | None = None,
    max_n: int = DEFAULT_MAX_N,
) -> dict[str, Any]:
    """Write 'running' state synchronously, then schedule the background
    fit job. Raises ``RuntimeError`` if a run is already in flight — the
    router surfaces this as HTTP 409 (same contract as
    ``item_scores.job.start_job`` / ``selection.job.start_job``).
    """
    if scope not in {'residual', 'cluster'}:
        raise ValueError(f"scope must be 'residual' or 'cluster', got {scope!r}")
    if scope == 'cluster' and cluster_id is None:
        raise ValueError("cluster_id is required when scope='cluster'")
    if _is_busy():
        raise RuntimeError('viz projection job already in progress')
    _ensure_dir()
    with contextlib.suppress(FileNotFoundError):
        _cancel_flag().unlink()
    with contextlib.suppress(FileNotFoundError):
        _heartbeat_file().unlink()

    job_id = uuid.uuid4().hex
    state = _JobState(
        job_id=job_id,
        status='running',
        scope=scope,
        cluster_id=cluster_id,
        started_at=time.time(),
    )
    _atomic_write(state)
    global _active_task  # noqa: PLW0603 - singleton task handle, mirrors item_scores.job
    _active_task = asyncio.create_task(
        run_projection_job(job_id, opensearch, scope=scope, cluster_id=cluster_id, max_n=max_n)
    )
    return state.to_dict()


def cancel_job() -> bool:
    if not _is_busy():
        return False
    _ensure_dir()
    _cancel_flag().touch()
    return True


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    xf = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(xf, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return xf / norms


def _subsample(
    ids: list[str], embeddings: np.ndarray, max_n: int, *, seed: int = VIZ_RANDOM_STATE
) -> tuple[list[str], np.ndarray]:
    """Deterministic random downsample to ``max_n`` rows. A no-op if the
    pool already fits. Random (not "first max_n") so the sample isn't
    biased toward whatever a scroll/slice happens to return first."""
    if len(ids) <= max_n:
        return ids, embeddings
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids), size=max_n, replace=False)
    idx.sort()
    return [ids[i] for i in idx.tolist()], embeddings[idx]


async def _fetch_pool(
    opensearch: AsyncOpenSearch,
    *,
    scope: str,
    cluster_id: int | None,
    max_n: int,
) -> tuple[list[str], np.ndarray]:
    """Reuse the two existing embedding-fetch helpers rather than a third
    scroll/PIT implementation (module docstring point 3)."""
    if scope == 'cluster':
        from src.services.curation.selection.pool_fetch import fetch_pool_embeddings

        query = {
            'bool': {
                'must': [
                    {'exists': {'field': EMBEDDING_FIELD}},
                    {'term': {'cluster_id': cluster_id}},
                ],
                'must_not': [{'term': {'test_holdout': True}}],
            }
        }
        ids, embeddings, truncated = await fetch_pool_embeddings(
            opensearch, ITEMS_INDEX, query, cap=max_n, embedding_field=EMBEDDING_FIELD
        )
        if truncated:
            # fetch_pool_embeddings returns an empty array when truncated
            # (its "give up on this cap" contract) -- refetch uncapped and
            # subsample ourselves instead, since a viz job is allowed to
            # take longer than an inline request.
            ids, embeddings, _ = await fetch_pool_embeddings(
                opensearch,
                ITEMS_INDEX,
                query,
                cap=10 * max_n,
                embedding_field=EMBEDDING_FIELD,
            )
            ids, embeddings = _subsample(ids, embeddings, max_n)
        return ids, embeddings

    # Remaining branch: scope is 'residual' (the only other value start_job accepts).
    from src.services.curation.clustering.embedding_reduce import (
        fetch_residual_v6_embeddings_parallel,
    )

    ids, embeddings = await fetch_residual_v6_embeddings_parallel(
        opensearch,
        extra_must=[{'bool': {'must_not': [{'term': {'test_holdout': True}}]}}],
    )
    ids, embeddings = _subsample(ids, embeddings, max_n)
    return ids, embeddings


def _build_reducer() -> Any:
    """Lazy ``import umap`` -- mirrors ``legacy_embedding_reduce._build_cpu_reducer``
    so importing this module (or running its non-fit tests) never requires
    ``umap-learn`` to be installed. CPU-only (no cuML branch): the plan's
    compute budget already treats 2-d UMAP as job-only/background
    (10-30 min CPU at 128k), and this is explicitly the lowest-priority
    phase -- a GPU path can be added later without touching the on-disk
    state format if the CPU wall-clock proves annoying in practice."""
    import umap

    return umap.UMAP(
        n_components=VIZ_N_COMPONENTS,
        n_neighbors=VIZ_N_NEIGHBORS,
        min_dist=VIZ_MIN_DIST,
        metric=VIZ_METRIC,
        random_state=VIZ_RANDOM_STATE,
    )


def _serialize_reducer(reducer: Any) -> bytes:
    import joblib

    buf = io.BytesIO()
    joblib.dump(reducer, buf)
    return buf.getvalue()


def _save_reducer_to_disk(reducer: Any) -> None:
    """Provenance-only persist -- see module docstring: nothing ever
    reloads this to `.transform()` new points, so a save failure must
    never fail the job (logged, not raised)."""
    try:
        p = Path(VIZ_STATE_JOBLIB_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_serialize_reducer(reducer))
    except Exception as exc:
        logger.warning('legacy_umap_viz_state_disk_save_failed', error=str(exc))


async def _save_run_metadata(
    opensearch: AsyncOpenSearch,
    *,
    scope: str,
    cluster_id: int | None,
    n_points: int,
    fitted_at: str,
) -> None:
    """Own metadata slot (``legacy_umap_viz_state``, state_id='current') --
    distinct from the retired clustering reducer's ``legacy_umap_state`` index.
    Metadata only (no pickled reducer blob) so this never risks the 100 MB
    OpenSearch request-body ceiling ``legacy_embedding_reduce.py`` had to work
    around for the (much larger, per-crop) clustering reducer blob."""
    body = {
        'state_id': 'current',
        'projection_version': VIZ_PROJECTION_VERSION,
        'scope': scope,
        'cluster_id': cluster_id,
        'n_points': n_points,
        'fitted_at': fitted_at,
        'n_components': VIZ_N_COMPONENTS,
        'metric': VIZ_METRIC,
    }
    try:
        await opensearch.index(index=UMAP_VIZ_STATE_INDEX, id='current', body=body, refresh=False)
    except Exception as exc:
        logger.warning('legacy_umap_viz_state_metadata_save_failed', error=str(exc))


async def _load_run_metadata(opensearch: AsyncOpenSearch) -> dict[str, Any] | None:
    try:
        resp = await opensearch.get(index=UMAP_VIZ_STATE_INDEX, id='current')
    except Exception as exc:
        logger.debug('legacy_umap_viz_state_metadata_not_found', error=str(exc))
        return None
    return resp.get('_source') or None


async def _bulk_write_coordinates(
    opensearch: AsyncOpenSearch,
    ids: list[str],
    xy: np.ndarray,
    *,
    chunk_size: int = 500,
) -> int:
    """Write ``viz_x``/``viz_y``/``viz_projection_version`` only -- never
    ``cluster_id``/``cluster_subid``/``cluster_distance`` (plan §8
    non-goal #3; guarded by
    ``tests/curation/test_embedding_viz.py::test_writes_never_include_cluster_fields``)."""
    index = ITEMS_INDEX
    n_written = 0
    for start in range(0, len(ids), chunk_size):
        chunk_ids = ids[start : start + chunk_size]
        chunk_xy = xy[start : start + chunk_size]
        bulk: list[dict[str, Any]] = []
        for crop_id, (x, y) in zip(chunk_ids, chunk_xy.tolist(), strict=True):
            bulk.append({'update': {'_index': index, '_id': crop_id}})
            bulk.append(
                {
                    'doc': {
                        'viz_x': float(x),
                        'viz_y': float(y),
                        'viz_projection_version': VIZ_PROJECTION_VERSION,
                    }
                }
            )
        if bulk:
            await opensearch.bulk(body=bulk, refresh=False)
            n_written += len(chunk_ids)
    return n_written


async def fit_projection(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    """The one function in this module that fits UMAP (note: fits only —
    despite the module writing coordinates elsewhere, this function does
    not itself touch OpenSearch; :func:`_bulk_write_coordinates` does that
    separately). Only ever called from :func:`run_projection_job` (module
    docstring point 2) -- never from ``legacy_viz.py``'s GET handler."""
    from datetime import UTC, datetime

    reducer = _build_reducer()
    xy_raw = await asyncio.to_thread(reducer.fit_transform, _l2_normalize(embeddings))
    xy = np.asarray(xy_raw, dtype=np.float32)
    _save_reducer_to_disk(reducer)
    fitted_at = datetime.now(UTC).isoformat()
    return xy, fitted_at


async def run_projection_job(
    job_id: str,
    opensearch: AsyncOpenSearch,
    *,
    scope: str,
    cluster_id: int | None,
    max_n: int,
) -> None:
    """Actual compute body: one pool fetch, one UMAP fit, one bulk write.
    A dedicated (non-underscore) symbol so tests can monkeypatch it
    wholesale, same reasoning as ``item_scores.job.run_scoring_job``."""
    state = _read_state()
    if state.job_id != job_id:
        return
    _touch_heartbeat()
    ticker = asyncio.create_task(_heartbeat_ticker())
    try:
        ids, embeddings = await _fetch_pool(
            opensearch, scope=scope, cluster_id=cluster_id, max_n=max_n
        )
        state.n_pool = len(ids)
        _atomic_write(state)
        _touch_heartbeat()

        if not ids:
            state.status = 'completed'
            state.n_written = 0
            state.finished_at = time.time()
            _atomic_write(state)
            return

        if is_cancelled():
            state.status = 'cancelled'
            state.finished_at = time.time()
            _atomic_write(state)
            return

        xy, fitted_at = await fit_projection(embeddings)
        _touch_heartbeat()

        if is_cancelled():
            state.status = 'cancelled'
            state.finished_at = time.time()
            _atomic_write(state)
            return

        n_written = await _bulk_write_coordinates(opensearch, ids, xy)
        await _save_run_metadata(
            opensearch,
            scope=scope,
            cluster_id=cluster_id,
            n_points=n_written,
            fitted_at=fitted_at,
        )

        state.status = 'completed'
        state.n_written = n_written
        state.projection_version = VIZ_PROJECTION_VERSION
        state.finished_at = time.time()
        _atomic_write(state)
    except asyncio.CancelledError:
        state.status = 'cancelled'
        state.finished_at = time.time()
        _atomic_write(state)
        raise
    except Exception as exc:
        logger.error('legacy_viz_projection_job_failed', job_id=job_id, error=str(exc))
        state.status = 'failed'
        state.error = str(exc)
        state.finished_at = time.time()
        _atomic_write(state)
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker


def _point_from_hit(h: dict[str, Any]) -> dict[str, Any]:
    src = h.get('_source') or {}
    return {
        'crop_id': h['_id'],
        'x': src.get('viz_x'),
        'y': src.get('viz_y'),
        'cluster_id': src.get('cluster_id'),
        'class_name': src.get('class_name'),
        'class_source': src.get('class_source'),
    }


async def get_cached_projection(
    opensearch: AsyncOpenSearch,
    *,
    cluster_id: int | None = None,
    class_id: int | None = None,
    max_points: int = 50_000,
) -> dict[str, Any]:
    """Serve **cached coordinates only** — imports nothing UMAP-related,
    calls no fit function, does one plain ``search`` over already-written
    ``viz_x``/``viz_y`` fields. This is the entire GET
    ``/legacy/viz/projection`` contract (module docstring point 2).

    Returns ``{'status': 'not_built'}`` when no projection has ever been
    fit. Otherwise ``{points, projection_version, fitted_at, stale}`` where
    ``stale`` is True iff there exist crops matching the requested scope
    (embedding present, not test_holdout) whose cached
    ``viz_projection_version`` doesn't match the latest fit's version --
    i.e. some in-scope crops are missing/outdated coordinates, the same
    "partial coverage is visible" philosophy ``/legacy/scores/coverage`` uses.
    """
    meta = await _load_run_metadata(opensearch)
    if meta is None:
        return {'status': 'not_built'}

    scope_must: list[dict[str, Any]] = [{'exists': {'field': EMBEDDING_FIELD}}]
    scope_must_not: list[dict[str, Any]] = [{'term': {'test_holdout': True}}]
    if cluster_id is not None:
        scope_must.append({'term': {'cluster_id': cluster_id}})
    if class_id is not None:
        scope_must.append({'term': {'class_id': class_id}})

    points_query = {
        'bool': {
            'filter': [*scope_must, {'exists': {'field': 'viz_x'}}],
            'must_not': scope_must_not,
        }
    }
    points: list[dict[str, Any]] = []
    search_after: list[Any] | None = None
    while len(points) < max_points:
        page_size = min(_VIZ_PROJECTION_PAGE_SIZE, max_points - len(points))
        body: dict[str, Any] = {
            'size': page_size,
            'query': points_query,
            '_source': ['viz_x', 'viz_y', 'cluster_id', 'class_name', 'class_source'],
            'sort': [{'crop_id': 'asc'}],
            'track_total_hits': False,
        }
        if search_after is not None:
            body['search_after'] = search_after
        resp = await opensearch.search(index=ITEMS_INDEX, body=body)
        hits = resp.get('hits', {}).get('hits') or []
        if not hits:
            break
        points.extend(_point_from_hit(h) for h in hits)
        if len(hits) < page_size:
            break
        search_after = hits[-1]['sort']

    stale = False
    try:
        missing_query = {
            'bool': {
                'must': scope_must,
                'must_not': [
                    *scope_must_not,
                    {'term': {'viz_projection_version': meta.get('projection_version', '')}},
                ],
            }
        }
        count_resp = await opensearch.count(index=ITEMS_INDEX, body={'query': missing_query})
        stale = int(count_resp.get('count', 0)) > 0
    except Exception as exc:
        logger.warning('legacy_viz_projection_staleness_check_failed', error=str(exc))

    return {
        'points': points,
        'projection_version': meta.get('projection_version'),
        'fitted_at': meta.get('fitted_at'),
        'stale': stale,
    }


__all__ = [
    'DEFAULT_MAX_N',
    'UMAP_VIZ_STATE_INDEX',
    'VIZ_METRIC',
    'VIZ_MIN_DIST',
    'VIZ_N_COMPONENTS',
    'VIZ_N_NEIGHBORS',
    'VIZ_PROJECTION_VERSION',
    'VIZ_RANDOM_STATE',
    'VIZ_STATE_JOBLIB_PATH',
    'cancel_job',
    'fit_projection',
    'get_cached_projection',
    'get_state',
    'is_cancelled',
    'reconcile_orphaned_jobs',
    'run_projection_job',
    'start_job',
]
