"""2-D UMAP projection for **visualization only**.

Non-negotiable design rules:

1. **Own persisted state slot.** Never touches
   ``umap_state.umap_state_joblib_path{,_cuml}()`` or the
   ``op_umap_state`` index — those belong to the *retired* clustering
   reducer (UMAP collapsed most of the residual pool into one
   mega-cluster and was retired for clustering). This module's state
   lives at ``umap_viz_state.joblib``
   (disk) and the ``op_umap_viz_state`` index — deliberately distinct
   names, never imported by/from ``embedding_reduce.py`` / ``umap_state.py`` /
   ``routers/curation/umap.py`` / ``clustering/orchestrator.py`` /
   ``clustering/methods/``.
2. **Never fits on a request path.** ``fit_projection`` (the only
   function importing ``umap`` / calling ``.fit_transform``) is reachable
   solely from :func:`run_projection_job`, queued only by
   :func:`start` (``POST /curation/viz/projection/rebuild``).
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
   :func:`src.clients.curation_opensearch.ensure_overlay_fields.ensure_items_viz_fields`).
   Never ``cluster_id`` / ``cluster_subid`` / ``cluster_distance`` (guarded
   by
   ``tests/curation/test_embedding_viz.py::test_writes_never_include_cluster_fields``).

Embedding fetch reuses the two existing helpers rather than a third
scroll/PIT-fetch implementation: ``scope='residual'``
delegates to
:func:`src.services.curation.clustering.residual_fetch.fetch_residual_embeddings_parallel`
(same pool + ``CONFIDENT_CLASS_SOURCES``/``class_excluded`` gate
``item_scores.job`` uses); ``scope='cluster'`` delegates to
:func:`src.services.curation.selection.pool_fetch.fetch_pool_embeddings`
with a ``cluster_id`` term filter (same building block the select router
uses). Both paths already exclude ``test_holdout=true`` crops.

The job is a queued ``embedding_viz`` job (one at a time per project); a Celery worker runs
:func:`run_projection_job`. UMAP is imported inside the task, never at module import. Every
rebuild does one full fresh ``fit_transform`` and writes coordinates; persisting the fitted
reducer is for provenance only, not a functional requirement.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.db.repos import settings as settings_repo
from src.db.session import async_session_factory, project_tx
from src.domain.embedding_spaces import Target
from src.domain.embedding_state import embedded_clause
from src.services.config_store.bound_project import bound_project_id
from src.services.curation.clustering.umap_state import reducer_stored, save_reducer
from src.services.curation.latest_job import common_state
from src.services.embedding_spaces.resolve import embedding_space
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    import uuid

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.schemas.jobs import JobView
    from src.tasks.runtime.context import JobContext


logger = get_logger(__name__)


# Run metadata is the project's ``projection_states`` row; the fitted reducer is a blob behind the
# storage seam under its own slot, never one of the clustering reducer's (umap_state.py).
VIZ_REDUCER_STATE_ID = 'viz'


# Viz-only UMAP hyperparameters. min_dist is higher than the clustering
# reducer's 0.0 (UMAP_MIN_DIST in pool_size.py) -- a viz scatter
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
# The integer the projection_states row stores; the items keep the string stamp above.
VIZ_PROJECTION_NUMBER = int(VIZ_PROJECTION_VERSION.rsplit('v', 1)[1])

KIND = 'embedding_viz'

# max_result_window is 10000; get_cached_projection() pages with search_after
# in chunks of this size instead of a single oversized `size: max_points`.

# Job pool cap. Residual-scope fetch has no built-in cap (unlike
# selection.pool_fetch's `cap` kwarg), so this module samples down to
# max_n *after* the fetch, deterministically (seeded) to avoid shard bias.
DEFAULT_MAX_N = 20_000


async def start(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    scope: str,
    cluster_id: int | None = None,
    max_n: int = DEFAULT_MAX_N,
) -> JobRow:
    """Queue the projection fit (409 ``job_busy`` while one is active; ``ValueError`` for a bad
    ``scope``)."""
    if scope not in {'residual', 'cluster'}:
        raise ValueError(f"scope must be 'residual' or 'cluster', got {scope!r}")
    if scope == 'cluster' and cluster_id is None:
        raise ValueError("cluster_id is required when scope='cluster'")
    started = await start_job(
        session,
        KIND,
        project,
        params={'scope': scope, 'cluster_id': cluster_id, 'max_n': max_n},
        requested_by='api',
    )
    return started.job


def state_dict(view: JobView | None) -> dict[str, Any]:
    """The ``/viz/projection/status`` wire of a job row."""
    params = (view.params if view else None) or {}
    progress = (view.progress if view else None) or {}
    result = (view.result if view else None) or {}
    return {
        **common_state(view),
        'scope': params.get('scope', ''),
        'cluster_id': params.get('cluster_id'),
        'n_pool': int(progress.get('total') or 0),
        'n_written': int(result.get('n_written') or 0),
        'projection_version': result.get('projection_version'),
    }


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
    embedding_field: str,
) -> tuple[list[str], np.ndarray]:
    """Reuse the two existing embedding-fetch helpers rather than a third
    scroll/PIT implementation (module docstring point 3)."""
    if scope == 'cluster':
        from src.services.curation.selection.pool_fetch import fetch_pool_embeddings

        query = {
            'bool': {
                'filter': [
                    embedded_clause(embedding_field),
                    {'term': {'cluster_id': cluster_id}},
                ],
                'must_not': [{'term': {'test_holdout': True}}],
            }
        }
        ids, embeddings, truncated = await fetch_pool_embeddings(
            opensearch, items_index(), query, cap=max_n, embedding_field=embedding_field
        )
        if truncated:
            # fetch_pool_embeddings returns an empty array when truncated
            # (its "give up on this cap" contract) -- refetch uncapped and
            # subsample ourselves instead, since a viz job is allowed to
            # take longer than an inline request.
            ids, embeddings, _ = await fetch_pool_embeddings(
                opensearch,
                items_index(),
                query,
                cap=10 * max_n,
                embedding_field=embedding_field,
            )
            ids, embeddings = _subsample(ids, embeddings, max_n)
        return ids, embeddings

    # Remaining branch: scope is 'residual' (the only other value start_job accepts).
    from src.services.curation.clustering.residual_fetch import fetch_residual_embeddings_parallel

    ids, embeddings = await fetch_residual_embeddings_parallel(
        opensearch,
        extra_must=[{'bool': {'must_not': [{'term': {'test_holdout': True}}]}}],
    )
    ids, embeddings = _subsample(ids, embeddings, max_n)
    return ids, embeddings


def _build_reducer() -> Any:
    """Lazy ``import umap`` -- mirrors ``clustering.embedding_reduce._build_cpu_reducer``
    so importing this module (or running its non-fit tests) never requires
    ``umap-learn`` to be installed. CPU-only (no cuML branch): the
    compute budget already treats 2-d UMAP as job-only/background
    (10-30 min CPU at 128k), and this is explicitly a low-priority
    concern -- a GPU path can be added later without touching the on-disk
    state format if the CPU wall-clock proves annoying in practice."""
    import umap

    return umap.UMAP(
        n_components=VIZ_N_COMPONENTS,
        n_neighbors=VIZ_N_NEIGHBORS,
        min_dist=VIZ_MIN_DIST,
        metric=VIZ_METRIC,
        random_state=VIZ_RANDOM_STATE,
    )


def _save_reducer(reducer: Any) -> None:
    """Provenance-only persist -- see module docstring: nothing ever
    reloads this to `.transform()` new points, so a save failure must
    never fail the job (logged, not raised)."""
    try:
        save_reducer(reducer, state_id=VIZ_REDUCER_STATE_ID)
    except OSError as exc:
        logger.warning('curation_umap_viz_state_save_failed', error=str(exc))


async def _save_run_metadata(
    *, scope: str, cluster_id: int | None, n_points: int, fitted_at: str
) -> None:
    """The project's current view state (one ``projection_states`` row): metadata only, the
    reducer blob is behind the storage seam and the row names its key."""
    project_id = await bound_project_id()
    try:
        async with async_session_factory()() as session, project_tx(session, project_id):
            await settings_repo.put_projection_state(
                session,
                project_id,
                projection_version=VIZ_PROJECTION_NUMBER,
                scope=scope,
                cluster_id=cluster_id,
                n_points=n_points,
                n_components=VIZ_N_COMPONENTS,
                metric=VIZ_METRIC,
                fitted_at=datetime.fromisoformat(fitted_at),
                reducer_storage_key=reducer_stored(state_id=VIZ_REDUCER_STATE_ID),
            )
    except SQLAlchemyError as exc:
        logger.warning('curation_umap_viz_state_metadata_save_failed', error=str(exc))


async def _bulk_write_coordinates(
    opensearch: AsyncOpenSearch,
    ids: list[str],
    xy: np.ndarray,
    *,
    chunk_size: int = 500,
) -> int:
    """Write ``viz_x``/``viz_y``/``viz_projection_version`` only -- never
    ``cluster_id``/``cluster_subid``/``cluster_distance`` (guarded by
    ``tests/curation/test_embedding_viz.py::test_writes_never_include_cluster_fields``)."""
    from src.clients.opensearch.bulk import failed_bulk_items

    index = items_index()
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
            resp = await opensearch.bulk(body=bulk, refresh=False)
            refused = len(failed_bulk_items(resp)) if isinstance(resp, dict) else 0
            n_written += len(chunk_ids) - refused
    if ids and n_written == 0:
        msg = f'no viz coordinate was written ({len(ids)} bulk items refused)'
        raise RuntimeError(msg)
    return n_written


async def fit_projection(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    """The one function in this module that fits UMAP (note: fits only —
    despite the module writing coordinates elsewhere, this function does
    not itself touch OpenSearch; :func:`_bulk_write_coordinates` does that
    separately). Only ever called from :func:`run_projection_job` (module
    docstring point 2) -- never from the reference embedding-viz router's GET handler."""
    from datetime import UTC

    reducer = _build_reducer()
    xy_raw = await asyncio.to_thread(reducer.fit_transform, _l2_normalize(embeddings))
    xy = np.asarray(xy_raw, dtype=np.float32)
    _save_reducer(reducer)
    fitted_at = datetime.now(UTC).isoformat()
    return xy, fitted_at


async def run_projection_job(ctx: JobContext, opensearch: AsyncOpenSearch) -> dict[str, Any]:
    """Actual compute body: one pool fetch, one UMAP fit, one bulk write. A cancel is checked
    before the fit and before the write (the fit itself cannot be interrupted)."""
    params = ctx.params
    embedding_field = (await embedding_space(Target.ITEM)).vector_field
    ids, embeddings = await _fetch_pool(
        opensearch,
        scope=params['scope'],
        cluster_id=params['cluster_id'],
        max_n=params['max_n'],
        embedding_field=embedding_field,
    )
    await ctx.progress(0, len(ids), 'fitting')
    if not ids:
        return {'n_written': 0}
    if ctx.cancelled():
        return {'cancelled': True}
    xy, fitted_at = await fit_projection(embeddings)
    if ctx.cancelled():
        return {'cancelled': True}
    n_written = await _bulk_write_coordinates(opensearch, ids, xy)
    await _save_run_metadata(
        scope=params['scope'],
        cluster_id=params['cluster_id'],
        n_points=n_written,
        fitted_at=fitted_at,
    )
    return {'n_written': n_written, 'projection_version': VIZ_PROJECTION_VERSION}


__all__ = [
    'DEFAULT_MAX_N',
    'KIND',
    'VIZ_METRIC',
    'VIZ_MIN_DIST',
    'VIZ_N_COMPONENTS',
    'VIZ_N_NEIGHBORS',
    'VIZ_PROJECTION_VERSION',
    'VIZ_RANDOM_STATE',
    'fit_projection',
    'run_projection_job',
    'start',
    'state_dict',
]
