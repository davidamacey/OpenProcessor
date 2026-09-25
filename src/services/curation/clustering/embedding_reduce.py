"""UMAP residual-pool reducer — visualization only (CM-7 doc fix).

Two genuinely separate things live in this module, and only one of
them touches real clustering:

1. :func:`fetch_residual_embeddings_parallel` — pulls v6/PE
   embeddings for residual crops (no ``cluster_id`` assigned). This
   part IS shared with the real clustering path:
   :func:`~src.services.curation.clustering.orchestrator.cluster_residuals`
   calls it directly for its embedding fetch.
2. UMAP reduction to 50 dimensions with a deterministic seed and
   ``cosine`` metric (CPU: ``umap-learn``; GPU: ``cuml.manifold.UMAP``
   when the worker container has a usable GPU and free VRAM — see
   :mod:`src.services.curation.clustering.backend`), plus persisting
   the fitted reducer (local joblib + OpenSearch ``op_umap_state``
   index, cache slots per-backend since sklearn/cuML pickle
   incompatible class instances).

**The UMAP-reduced output from (2) is never fed into AHC or IVF.**
``cluster_residuals`` clusters the raw, L2-normalized 1024-d embeddings
fetched in (1) directly — that's also what makes the AHC
``distance_threshold=0.25`` cosine cut dimensionally correct against
the source vectors, not a reduced projection. The only caller of the
UMAP-reduce path is ``POST /curation/cluster/umap/rebuild``
(:mod:`src.routers.curation_umap`), which exists purely to feed the
labeler's 2D/3D scatter visualization. Any design note claiming "UMAP
feeds the clustering pipeline" is wrong; correct it if you find one.

Defaults are tuned for the reference embedding pipeline (a private
design doc's Appendix C.12
of the redesign plan); override the embedding field via
``OP_RESIDUAL_EMBEDDING_FIELD`` if experimenting.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.clustering.backend import (
    BackendInfo,
    detect_cluster_backend,
    free_gpu_blocks,
    gpu_used_vram_mb,
)
from src.services.curation.ingest_class_sources import confident_class_sources


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


UMAP_STATE_INDEX = get_curation_config().umap_state_index
ITEMS_INDEX = get_curation_config().items_index

# State dir shared with the VLM worker via the GPU arbiter pause
# sentinel. Deployment-configurable via CurationConfig.state_dir rather
# than a dedicated env var, so it stays consistent with every other
# persisted-state path in the curation subsystem.
_STATE_DIR = get_curation_config().state_dir
UMAP_STATE_JOBLIB_PATH = str(Path(_STATE_DIR) / 'umap_state.joblib')
UMAP_STATE_JOBLIB_PATH_CUML = str(Path(_STATE_DIR) / 'umap_state_cuml.joblib')

# UMAP hyperparameters — Appendix C.12 of the reference redesign plan.
UMAP_N_COMPONENTS = 50
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
UMAP_METRIC = 'cosine'
UMAP_RANDOM_STATE = 42

# Embedding field to reduce. ``pe_embedding`` is PE-Core-L14-336's
# foundation visual encoder (1024-d), trained on broad web imagery —
# better at grouping out-of-distribution vehicles by semantic similarity
# than ``backbone_embedding`` (the v6 classifier's penultimate layer, which
# clusters by framing/lighting outside its confident range). The
# 2026-05-18 visual audit picked pe_embedding as the residual default.
RESIDUAL_EMBEDDING_FIELD = os.environ.get('OP_RESIDUAL_EMBEDDING_FIELD', 'pe_embedding')

# Crops with these class_source values are *confidently* labeled and
# must NEVER be pulled into the residual pool — clustering would
# overwrite their cluster_id=class_id mapping with a candidate id and
# destroy the class-cluster grouping (regression 2026-05-23: an IVF
# run on the unfiltered query overwrote ~129k labeled crops' cluster
# ids before being cancelled).
#
# Specifically (src.services.curation.ingest_class_sources, derived from
# the configured ingest profiles -- resolved once at import like other
# OP_* config):
#   {secondary}_model — the ingest classifier labeled it above its floor
#                       ({primary}_model too when the primary assigns_class)
#   vlm               — the VLM matched a registered class name
#   human             — human-validated label
#
# Everything else (low-conf / unlabeled proposals, vlm_unmatched,
# vlm_new_class_pending, null) is residual and goes into the cluster.
CONFIDENT_CLASS_SOURCES = confident_class_sources()

UMapMode = Literal['transform', 'refit']


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _state_paths_for(backend: str) -> tuple[str, str]:
    """Return ``(joblib_path, opensearch_state_id)`` for the given backend.

    Per-backend cache slots so a worker flipping CPU<->GPU doesn't try
    to unpickle a sklearn UMAP class as a cuML UMAP class (or vice
    versa). The ``current`` OpenSearch state id is the legacy sklearn
    slot; the GPU slot is ``current_cuml``.
    """
    if backend == 'gpu':
        return UMAP_STATE_JOBLIB_PATH_CUML, 'current_cuml'
    return UMAP_STATE_JOBLIB_PATH, 'current'


async def fetch_residual_embeddings(
    client: AsyncOpenSearch,
    *,
    include_candidate_clusters: bool = False,
    candidate_cluster_id_min: int | None = None,
    extra_must: list[dict[str, Any]] | None = None,
    progress: Any | None = None,
) -> tuple[list[str], np.ndarray]:
    """Pull unvalidated crops with embeddings for the AHC residual pool.

    Two pools, selected by ``include_candidate_clusters``:

    * **Default (False)** — crops with ``RESIDUAL_EMBEDDING_FIELD`` set,
      ``class_validated`` not true, AND either no ``cluster_id`` yet
      OR a ``cluster_id`` below ``candidate_cluster_id_min``. The intent
      is "everything unvalidated that isn't already in a candidate
      cluster" — fresh items (no cluster_id) plus items still sitting
      in v6 class buckets (cluster_id < candidate_cluster_id_min).
      Existing candidate clusters are left alone so they aren't churned
      every run.
    * **Broad (True)** — same, but candidate-bucketed crops are also
      included. Use this when the operator wants to re-pool candidate
      clusters so smaller ones can merge into bigger ones.

    Despite the name's ``v6_`` prefix (kept for back-compat with the
    pipeline stages), the actual field read is governed by
    :py:data:`RESIDUAL_EMBEDDING_FIELD` — defaults to PE embeddings.

    When ``progress`` is supplied, the scroll loop updates
    ``processed`` after each page and checks the cancel sentinel so an
    operator-initiated cancel takes effect within a single scroll
    batch (~50 ms) instead of waiting for the full fetch to finish.
    """
    filt: list[dict[str, Any]] = [{'exists': {'field': RESIDUAL_EMBEDDING_FIELD}}]
    # Exclude confidently-labeled crops from the residual pool. The
    # previous filter (class_validated != true) only caught the 102
    # human-validated rows because item_model and vlm writers don't
    # set class_validated; the 2026-05-23 IVF run pulled all 347k
    # crops including 219k labeled ones and started overwriting their
    # cluster_id=class_id mappings before cancel. CONFIDENT_CLASS_SOURCES
    # is the right gate: only crops with no confident class_source go
    # into clustering.
    must_not: list[dict[str, Any]] = [
        {'term': {'class_validated': True}},
        {'terms': {'class_source': list(CONFIDENT_CLASS_SOURCES)}},
        # Human-ignored crops (blurry / unidentifiable) stay out of the
        # residual pool so they're never re-clustered.
        {'term': {'class_excluded': True}},
    ]
    if not include_candidate_clusters and candidate_cluster_id_min is not None:
        # Include items with no cluster_id OR cluster_id below the
        # candidate threshold. Items already in a candidate cluster
        # (>= threshold) are excluded so they aren't re-pooled.
        filt.append(
            {
                'bool': {
                    'should': [
                        {'bool': {'must_not': [{'exists': {'field': 'cluster_id'}}]}},
                        {'range': {'cluster_id': {'lt': candidate_cluster_id_min}}},
                    ],
                    'minimum_should_match': 1,
                },
            },
        )
    if extra_must:
        # Primary-subject clustering gate (rank / blur). Narrows the pool
        # to the crops that should train centroids + be assigned.
        filt.extend(extra_must)

    # Pre-count so the progress bar has a real total. A separate count
    # query is cheap (no _source, no scroll) and lets the dashboard show
    # 12,000 / 90,000 instead of "working…" during the scroll.
    total_estimate = 0
    if progress is not None:
        try:
            count_resp = await client.count(
                index=ITEMS_INDEX,
                body={'query': {'bool': {'filter': filt, 'must_not': must_not}}},
            )
            total_estimate = int(count_resp.get('count', 0))
            progress.update(processed=0, total=total_estimate)
        except Exception as exc:
            logger.debug('curation_embedding_count_failed', error=str(exc))

    body: dict[str, Any] = {
        'size': 1000,
        '_source': [RESIDUAL_EMBEDDING_FIELD],
        'query': {'bool': {'filter': filt, 'must_not': must_not}},
    }
    ids: list[str] = []
    embs: list[np.ndarray] = []
    resp = await client.search(index=ITEMS_INDEX, body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    try:
        while True:
            hits = resp['hits']['hits']
            if not hits:
                break
            for h in hits:
                src = h.get('_source') or {}
                emb = src.get(RESIDUAL_EMBEDDING_FIELD)
                if emb is None:
                    continue
                ids.append(h['_id'])
                embs.append(np.asarray(emb, dtype=np.float32))
            if progress is not None:
                # Scroll-loop checkpoint: emit progress and honour cancel
                # after each page. Per-page is ~50 ms so a cancel is felt
                # within a fraction of a second of the click.
                progress.update(processed=len(ids), total=total_estimate or len(ids))
                progress.raise_if_cancelled()
            if not scroll_id:
                break
            resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = resp.get('_scroll_id')
    finally:
        if scroll_id:
            try:
                await client.clear_scroll(scroll_id=scroll_id)
            except Exception as exc:
                logger.debug('curation_embedding_clear_scroll_failed', error=str(exc))

    if not embs:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, _normalize_rows(np.vstack(embs))


# ============================================================================
# PIT + sliced parallel fetch — N-way concurrent variant of the scroll
# loop above. ~8x faster on 347k docs (10 min -> ~1.5 min) by letting
# OpenSearch coordinate N async slices over a frozen Point-In-Time view.
#
# Use this for cluster_residuals; keep the scroll variant for callers
# that don't have a sortable index or want simpler semantics.
# ============================================================================


PARALLEL_FETCH_DEFAULT_SLICES = 8
PARALLEL_FETCH_PAGE_SIZE = 2000
PARALLEL_FETCH_PIT_KEEPALIVE = '5m'


async def fetch_residual_embeddings_parallel(
    client: AsyncOpenSearch,
    *,
    include_candidate_clusters: bool = False,
    candidate_cluster_id_min: int | None = None,
    extra_must: list[dict[str, Any]] | None = None,
    n_slices: int = PARALLEL_FETCH_DEFAULT_SLICES,
    page_size: int = PARALLEL_FETCH_PAGE_SIZE,
    progress: Any | None = None,
) -> tuple[list[str], np.ndarray]:
    """Same shape as :func:`fetch_residual_embeddings`, PIT + slice impl.

    Freezes a Point-In-Time view of the index and dispatches
    ``n_slices`` parallel async scrolls (each owning ``slice {id: i,
    max: n_slices}``). OpenSearch partitions by shard+doc-id so the
    slices return disjoint result sets. PIT keeps results stable even
    if the ingest path is writing concurrently.

    Falls back to the sequential scroll-based fetcher on any error
    (older OpenSearch, missing _shard_doc sort support, etc.) so the
    pipeline never breaks on a fetch-layer issue.
    """
    filt: list[dict[str, Any]] = [{'exists': {'field': RESIDUAL_EMBEDDING_FIELD}}]
    # Same residual-pool gate as the scroll variant — exclude
    # confidently-labeled crops so clustering only touches the truly
    # residual cohort. See CONFIDENT_CLASS_SOURCES for the rule.
    must_not: list[dict[str, Any]] = [
        {'term': {'class_validated': True}},
        {'terms': {'class_source': list(CONFIDENT_CLASS_SOURCES)}},
        # Human-ignored crops (blurry / unidentifiable) stay out of the
        # residual pool so they're never re-clustered.
        {'term': {'class_excluded': True}},
    ]
    if not include_candidate_clusters and candidate_cluster_id_min is not None:
        filt.append(
            {
                'bool': {
                    'should': [
                        {'bool': {'must_not': [{'exists': {'field': 'cluster_id'}}]}},
                        {'range': {'cluster_id': {'lt': candidate_cluster_id_min}}},
                    ],
                    'minimum_should_match': 1,
                },
            },
        )
    if extra_must:
        # Primary-subject clustering gate (rank / blur).
        filt.extend(extra_must)
    query = {'bool': {'filter': filt, 'must_not': must_not}}

    # Pre-count for progress.
    total_estimate = 0
    if progress is not None:
        try:
            count_resp = await client.count(index=ITEMS_INDEX, body={'query': query})
            total_estimate = int(count_resp.get('count', 0))
            progress.update(processed=0, total=total_estimate)
        except Exception as exc:
            logger.debug('curation_embedding_count_failed', error=str(exc))

    # Create a Point-In-Time. Without PIT we'd need scroll cursors per
    # slice -- PIT is the supported pattern for sliced parallel reads.
    try:
        pit_resp = await client.create_pit(
            index=ITEMS_INDEX,
            keep_alive=PARALLEL_FETCH_PIT_KEEPALIVE,
        )
        pit_id = pit_resp.get('pit_id') or pit_resp.get('pit')
        if not pit_id:
            raise RuntimeError(f'create_pit returned no pit_id: {pit_resp!r}')
    except Exception as exc:
        logger.warning('curation_embedding_pit_unavailable_fallback_scroll', error=str(exc))
        return await fetch_residual_embeddings(
            client,
            include_candidate_clusters=include_candidate_clusters,
            candidate_cluster_id_min=candidate_cluster_id_min,
            extra_must=extra_must,
            progress=progress,
        )

    # Shared progress counter across slices; integer mutation under the
    # GIL is atomic enough for an int-bucket counter.
    counter = {'n': 0}
    lock = asyncio.Lock()

    async def _emit_progress(added: int) -> None:
        if progress is None:
            return
        async with lock:
            counter['n'] += added
            progress.update(processed=counter['n'], total=total_estimate or counter['n'])
            progress.raise_if_cancelled()

    async def _fetch_one_slice(slice_id: int) -> tuple[list[str], list[np.ndarray]]:
        ids_local: list[str] = []
        embs_local: list[np.ndarray] = []
        search_after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                'size': page_size,
                '_source': [RESIDUAL_EMBEDDING_FIELD],
                'query': query,
                'pit': {'id': pit_id, 'keep_alive': PARALLEL_FETCH_PIT_KEEPALIVE},
                # _shard_doc is a synthetic sort key OpenSearch exposes
                # specifically for PIT pagination; doesn't require a
                # mapping change. Same sort across all slices keeps the
                # cursors comparable.
                'sort': [{'_shard_doc': 'asc'}],
            }
            if n_slices > 1:
                body['slice'] = {'id': slice_id, 'max': n_slices}
            if search_after is not None:
                body['search_after'] = search_after

            resp = await client.search(body=body)
            hits = resp.get('hits', {}).get('hits') or []
            if not hits:
                break
            for h in hits:
                src = h.get('_source') or {}
                emb = src.get(RESIDUAL_EMBEDDING_FIELD)
                if emb is None:
                    continue
                ids_local.append(h['_id'])
                embs_local.append(np.asarray(emb, dtype=np.float32))
            await _emit_progress(len(hits))
            # search_after cursor = sort key of the last hit. Required
            # to continue paging within this slice.
            last_sort = hits[-1].get('sort')
            if not last_sort:
                break
            search_after = last_sort
        return ids_local, embs_local

    try:
        slice_results = await asyncio.gather(
            *(_fetch_one_slice(i) for i in range(n_slices)),
        )
    finally:
        try:
            await client.delete_pit(body={'pit_id': [pit_id]})
        except Exception as exc:
            logger.debug('curation_embedding_pit_delete_failed', error=str(exc))

    ids: list[str] = []
    embs: list[np.ndarray] = []
    for sl_ids, sl_embs in slice_results:
        ids.extend(sl_ids)
        embs.extend(sl_embs)

    if not embs:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, _normalize_rows(np.vstack(embs))


def _serialize_reducer(reducer: Any) -> bytes:
    import joblib

    buf = io.BytesIO()
    joblib.dump(reducer, buf)
    return buf.getvalue()


def _deserialize_reducer(blob: bytes) -> Any:
    import joblib

    return joblib.load(io.BytesIO(blob))


def _save_umap_state_to_disk(reducer: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_serialize_reducer(reducer))


def _load_umap_state_from_disk(path: str) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return _deserialize_reducer(p.read_bytes())
    except Exception as exc:
        # Stale or backend-incompatible pickle — log + refit. Don't
        # crash the run because last week's reducer can't unpickle in
        # the new container.
        logger.warning('curation_umap_state_disk_load_failed', path=path, error=str(exc))
        return None


# OpenSearch's default HTTP request limit is 100 MB. cuML's pickled
# UMAP retains the training data on the GPU (cuml#5818), which inflates
# the joblib blob to ~120 MB+ at 90k x 1024. Skip the OpenSearch
# fallback when the blob is over this threshold — the disk-cached copy
# is still written, and a refit on a fresh container is acceptable.
_OPENSEARCH_PERSIST_MAX_BYTES = 60 * 1024 * 1024


async def _save_umap_state_to_opensearch(
    client: AsyncOpenSearch, reducer: Any, *, state_id: str
) -> None:
    blob = _serialize_reducer(reducer)
    if len(blob) > _OPENSEARCH_PERSIST_MAX_BYTES:
        logger.info(
            'curation_umap_state_skip_opensearch_persist',
            reason='blob_too_large',
            blob_bytes=len(blob),
            limit_bytes=_OPENSEARCH_PERSIST_MAX_BYTES,
            state_id=state_id,
        )
        return
    body = {
        'state_id': state_id,
        'reducer_b64': base64.b64encode(blob).decode('ascii'),
        'n_components': UMAP_N_COMPONENTS,
        'metric': UMAP_METRIC,
    }
    await client.index(index=UMAP_STATE_INDEX, id=state_id, body=body, refresh=False)


async def _load_umap_state_from_opensearch(client: AsyncOpenSearch, *, state_id: str) -> Any | None:
    try:
        resp = await client.get(index=UMAP_STATE_INDEX, id=state_id)
    except Exception as exc:
        logger.debug('curation_umap_state_not_found', state_id=state_id, error=str(exc))
        return None
    src = resp.get('_source') or {}
    blob_b64 = src.get('reducer_b64')
    if not blob_b64:
        return None
    try:
        return _deserialize_reducer(base64.b64decode(blob_b64))
    except Exception as exc:
        logger.warning('curation_umap_state_os_load_failed', state_id=state_id, error=str(exc))
        return None


def _build_cpu_reducer() -> Any:
    """Construct a freshly-parameterised sklearn-side ``umap.UMAP``."""
    import umap

    return umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )


def _build_gpu_reducer(build_algo: str) -> Any:
    """Construct a freshly-parameterised ``cuml.manifold.UMAP``.

    ``build_algo`` picks the kNN construction strategy:
    * ``'nn_descent'`` — fast, non-deterministic, lower VRAM peak.
    * ``'brute_force_knn'`` — deterministic, ~2x VRAM peak.

    Detection picks based on the free-VRAM probe in
    :mod:`cluster_backend`; this function just consumes the choice.
    """
    import cuml  # type: ignore[import-not-found]

    kwargs: dict[str, Any] = {
        'n_components': UMAP_N_COMPONENTS,
        'n_neighbors': UMAP_N_NEIGHBORS,
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
            reducer = _build_gpu_reducer(backend.build_algo)
            logger.info(
                'curation_umap_fit_gpu',
                build_algo=backend.build_algo,
                free_vram_mb=backend.free_vram_mb,
                n=len(embeddings),
            )
        else:
            reducer = _build_cpu_reducer()
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

    Exposed via ``POST /curation/cluster/umap/rebuild``. CM-7: this is
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


__all__ = [
    'ITEMS_INDEX',
    'RESIDUAL_EMBEDDING_FIELD',
    'UMAP_METRIC',
    'UMAP_MIN_DIST',
    'UMAP_N_COMPONENTS',
    'UMAP_N_NEIGHBORS',
    'UMAP_RANDOM_STATE',
    'UMAP_STATE_INDEX',
    'UMAP_STATE_JOBLIB_PATH',
    'UMAP_STATE_JOBLIB_PATH_CUML',
    'fetch_residual_embeddings',
    'get_or_fit_reducer',
    'umap_rebuild',
]
