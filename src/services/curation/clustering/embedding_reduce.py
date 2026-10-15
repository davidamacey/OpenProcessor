"""UMAP residual-pool reducer fit and rebuild — visualization only.

Two genuinely separate things are involved in residual clustering, and only
one of them touches real clustering:

1. Fetching classifier/PE embeddings for residual crops, in
   :mod:`src.services.curation.clustering.residual_fetch`. That part IS
   shared with the real clustering path:
   :func:`~src.services.curation.clustering.orchestrator.cluster_residuals`
   calls it directly for its embedding fetch.
2. UMAP reduction to 50 dimensions with a deterministic seed and
   ``cosine`` metric (CPU: ``umap-learn``; GPU: ``cuml.manifold.UMAP``
   when the worker container has a usable GPU and free VRAM — see
   :mod:`src.services.curation.clustering.backend`), in this module, plus
   persisting the fitted reducer
   (:mod:`src.services.curation.clustering.umap_state`).

**The UMAP-reduced output from (2) is never fed into AHC or IVF.**
``cluster_residuals`` clusters the raw, L2-normalized 1024-d embeddings
fetched in (1) directly — that's also what makes the AHC
``distance_threshold=0.25`` cosine cut dimensionally correct against
the source vectors, not a reduced projection. The only caller of the
UMAP-reduce path is ``POST /curation/cluster/umap/rebuild``
(:mod:`src.routers.curation.umap`), which exists purely to feed the
labeler's 2D/3D scatter visualization. Any design note claiming "UMAP
feeds the clustering pipeline" is wrong; correct it if you find one.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.core.logging import get_logger
from src.services.curation.clustering.backend import (
    BackendInfo,
    detect_cluster_backend,
    free_gpu_blocks,
    gpu_used_vram_mb,
)
from src.services.curation.clustering.pool_size import (
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_RANDOM_STATE,
    require_min_items,
    umap_shape,
)
from src.services.curation.clustering.residual_fetch import fetch_residual_embeddings
from src.services.curation.clustering.umap_state import (
    _load_umap_state_from_disk,
    _load_umap_state_from_opensearch,
    _save_umap_state_to_disk,
    _save_umap_state_to_opensearch,
    _state_paths_for,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


UMapMode = Literal['transform', 'refit']


def _build_cpu_reducer(n_items: int) -> Any:
    """Construct a freshly-parameterised sklearn-side ``umap.UMAP``."""
    import umap

    return umap.UMAP(
        **umap_shape(n_items),
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )


def _build_gpu_reducer(build_algo: str, n_items: int) -> Any:
    """Construct a freshly-parameterised ``cuml.manifold.UMAP``.

    ``build_algo`` picks the kNN construction strategy:
    * ``'nn_descent'`` — fast, non-deterministic, lower VRAM peak.
    * ``'brute_force_knn'`` — deterministic, ~2x VRAM peak.

    Detection picks based on the free-VRAM probe in
    :mod:`cluster_backend`; this function just consumes the choice.
    """
    import cuml

    shape = umap_shape(n_items)
    kwargs: dict[str, Any] = {
        'n_components': shape['n_components'],
        'n_neighbors': shape['n_neighbors'],
        'min_dist': UMAP_MIN_DIST,
        'metric': UMAP_METRIC,
        'random_state': UMAP_RANDOM_STATE,
        'build_algo': build_algo,
    }
    return cuml.manifold.UMAP(**kwargs)


def _to_numpy_float32(arr: Any) -> np.ndarray:
    """Coerce a cuML output (cupy/cudf) to host float32. No-op for ndarrays."""
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32, copy=False)
    # cupy ndarrays expose .get(); cudf via .to_numpy().
    for meth in ('get', 'to_numpy'):
        fn = getattr(arr, meth, None)
        if not callable(fn):
            continue
        try:
            return np.asarray(fn(), dtype=np.float32)
        except Exception as exc:  # nosec B110 — try alternates
            logger.debug('curation_to_numpy_method_failed', method=meth, error=str(exc))
    return np.asarray(arr, dtype=np.float32)


async def get_or_fit_reducer(
    client: AsyncOpenSearch,
    embeddings: np.ndarray,
    *,
    mode: UMapMode,
    progress: Any | None = None,
) -> tuple[Any, np.ndarray, bool, BackendInfo]:
    """Return ``(reducer, reduced_embeddings, refit_happened, backend)``.

    ``mode='transform'``: load cached reducer from joblib first, then
    OpenSearch; refit only if nothing is cached.
    ``mode='refit'``: always re-fit on the current pool and overwrite
    both persisted copies.

    The backend (GPU vs CPU) is detected per call so a worker container
    that loses its GPU (e.g. ``make gpu-free``) falls back to sklearn
    on the next run without a restart. Per-backend cache slots prevent
    cross-backend unpickle errors.
    """
    # asyncio.to_thread: detect_cluster_backend()'s docstring claims "a
    # few microseconds," but cupy.cuda.runtime.getDeviceCount() has been
    # observed to block 30+ seconds when hitting a driver-mismatch error
    # in a container with no GPU device nodes at all (this worker is
    # deliberately CPU-only) -- long enough to starve the sibling
    # heartbeat coroutine and trigger a false "worker heartbeat stale"
    # job failure. Off the event loop, the wait is harmless.
    backend = await asyncio.to_thread(detect_cluster_backend)
    joblib_path, os_state_id = _state_paths_for(backend.name)

    refit_happened = False
    reducer: Any | None = None

    if mode == 'transform':
        reducer = _load_umap_state_from_disk(joblib_path)
        if reducer is None:
            reducer = await _load_umap_state_from_opensearch(client, state_id=os_state_id)

    # Sample VRAM before the heavy work so the dashboard's peak counter
    # reflects the run's hot loop, not just the post-fit residual.
    if progress is not None and backend.name == 'gpu':
        progress.record_peak_vram(gpu_used_vram_mb())

    if reducer is None or mode == 'refit':
        if backend.name == 'gpu' and backend.build_algo is not None:
            reducer = _build_gpu_reducer(backend.build_algo, len(embeddings))
            logger.info(
                'curation_umap_fit_gpu',
                build_algo=backend.build_algo,
                free_vram_mb=backend.free_vram_mb,
                n=len(embeddings),
            )
        else:
            reducer = _build_cpu_reducer(len(embeddings))
            logger.info('curation_umap_fit_cpu', n=len(embeddings))
        try:
            reduced_raw = await asyncio.to_thread(reducer.fit_transform, embeddings)
        finally:
            if backend.name == 'gpu':
                # cuml#4068: free residual pool to avoid VRAM creep
                # across reclusters in a long-lived worker.
                free_gpu_blocks()
                if progress is not None:
                    progress.record_peak_vram(gpu_used_vram_mb())
        reduced = _to_numpy_float32(reduced_raw)
        _save_umap_state_to_disk(reducer, joblib_path)
        try:
            await _save_umap_state_to_opensearch(client, reducer, state_id=os_state_id)
        except Exception as exc:
            logger.warning('curation_umap_state_persist_failed', error=str(exc))
        refit_happened = True
    else:
        try:
            reduced_raw = await asyncio.to_thread(reducer.transform, embeddings)
        finally:
            if backend.name == 'gpu':
                free_gpu_blocks()
                if progress is not None:
                    progress.record_peak_vram(gpu_used_vram_mb())
        reduced = _to_numpy_float32(reduced_raw)

    return reducer, reduced, refit_happened, backend


async def umap_rebuild(client: AsyncOpenSearch) -> dict[str, Any]:
    """Force a UMAP refit over the current residual pool.

    Exposed via ``POST /curation/cluster/umap/rebuild``. This is
    visualization-only -- it refits the manifold cache the labeler's
    scatter view reads, and has no effect on the next auto-label run.
    AHC/IVF cluster the raw 1024-d embeddings directly and never
    consume this module's UMAP-reduced output; see the module
    docstring.
    """
    ids, embeddings = await fetch_residual_embeddings(client)
    if len(ids) == 0:
        return {
            'status': 'no_residuals',
            'n_residuals': 0,
            'refit': False,
        }
    require_min_items(len(ids))
    _, _reduced, refit_happened, backend = await get_or_fit_reducer(
        client, embeddings, mode='refit'
    )
    return {
        'status': 'success',
        'n_residuals': len(ids),
        'refit': bool(refit_happened),
        'backend': backend.name,
        'backend_detail': backend.detail,
    }
