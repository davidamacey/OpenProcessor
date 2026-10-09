"""
Curation clustering orchestrator — runs the residual-pool clusterer.

The residual-pool algorithm implementations live in
:py:mod:`src.services.curation.clustering.methods` behind a small registry
(IVF by default, AHC / HDBSCAN as alternatives). The sibling modules hold
the other concerns: ``refine`` (per-cluster AHC), ``retrain_policy`` (when
to retrain IVF centroids), ``residual_gate`` (primary-subject gate) and
``cluster_write_guard`` (guarded bulk writes).

Public surface:

1. :py:func:`cluster_residuals` — residual-pool clusterer; dispatches
   through :py:func:`cluster_methods.get_method`.
2. :py:func:`assign_only_residuals` — re-sort against persisted centroids.

Skip-rule: residual pool < 32 residuals → no-op (return empty summary).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.curation import items_index
from src.core.logging import get_logger
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.clustering.cluster_write_guard import (
    _guarded_class_cluster_write,
    _log_bulk_write_errors,
)
from src.services.curation.clustering.last_run import record_last_run
from src.services.curation.clustering.pool_size import MIN_RESIDUALS_FOR_CLUSTERING
from src.services.curation.clustering.residual_gate import (
    _park_gated_residuals,
    gate_must_clauses,
    residual_gate_coverage,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


async def cluster_residuals(
    client: AsyncOpenSearch,
    *,
    recluster_unvalidated: bool = False,
    clustering_method: str | None = None,
    gate_max_rank: int | None = None,
    gate_min_blur_ratio: float | None = None,
    n_clusters: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Cluster the residual pool of unvalidated PE embeddings.

    Dispatches to a :class:`ClusterMethod` from
    :py:mod:`src.services.curation.clustering.methods`:

    * ``"ivf"`` (default) — FAISS k-means partitioning into a fixed K
      buckets, trained on a bounded sample and persisted so ingest +
      assign_only reuse the centroids. No ``-1`` noise; every crop gets
      a bucket. Chosen because the item-embedding manifold is
      continuously dense (no density gaps for HDBSCAN, no geometry for
      UMAP to preserve).
    * ``"ahc"`` — sklearn AgglomerativeClustering complete-linkage on a
      sparse cosine kNN graph. Same algorithm as the refine endpoint;
      retained as a fallback. Don't use on large n — the C-level
      merge loop starves the worker heartbeat coroutine.
    * ``"hdbscan"`` — cuML GPU HDBSCAN; dormant. Collapses on this data
      (one mega-cluster or 100% noise) but kept for a future curated
      subset that has actual density structure.

    Pipeline:

    1. Fetch embeddings for unvalidated crops. Two pool modes:

       * default — items not yet in a candidate cluster.
       * ``recluster_unvalidated=True`` — also pulls items already in
         candidate clusters so smaller candidates can fuse.

    2. Dispatch to the chosen :class:`ClusterMethod`.
    3. Add the ``RESIDUAL_CLUSTER_ID_OFFSET`` to the method's raw labels
       so candidate ids never collide with the class-id namespace
       (0..80). The raw label order is written as-is (no size-based
       renumbering) so this matches :func:`assign_only_residuals` and
       the ingest-time assign path, which both write
       ``RESIDUAL_CLUSTER_ID_OFFSET + raw_centroid_index`` straight from
       :class:`IVFCentroidStore`. A prior version renumbered so
       cluster_id=0 was always the largest bucket, but that renumbering
       was applied only here — never persisted to the centroid store,
       and never applied by ingest/assign_only — so the same crop could
       get a different cluster_id depending on which code path last
       clustered it. If a largest-first ordering is needed again, it
       must be a derived sort (e.g. by cluster size, computed from the
       live data) rather than baked into the id itself.
    4. Bulk-write back to OpenSearch in chunks of 2000.
    """
    from src.services.curation.clustering import embedding_reduce
    from src.services.curation.clustering.backend import detect_cluster_backend
    from src.services.curation.clustering.methods import DEFAULT_METHOD, get_method
    from src.services.curation.strategy_registry import resolve_effective_default

    # DEFAULT_METHOD stays the ultimate fallback (resolve_effective_default
    # falls back to it internally too) -- an explicit ?clustering_method
    # always wins over any shared-settings override, same precedence every
    # other real endpoint's omitted-param resolution uses.
    if clustering_method:
        method_name = clustering_method.lower()
    else:
        resolved_default = await resolve_effective_default('cluster', client)
        method_name = (resolved_default or DEFAULT_METHOD).lower()
    mode_label = 'recluster_unvalidated' if recluster_unvalidated else 'strict_residuals'

    # Primary-subject clustering gate (optional). When set, train + assign
    # only the largest, clear crops; park the rest. Block on insufficient
    # field coverage so a half-backfilled pool can't silently empty itself.
    gate_clauses = gate_must_clauses(gate_max_rank, gate_min_blur_ratio)
    gate_active = bool(gate_clauses)
    if gate_active:
        coverage = await residual_gate_coverage(
            client, max_rank=gate_max_rank, min_blur_ratio=gate_min_blur_ratio
        )
        if not coverage['sufficient']:
            return {
                'status': 'gate_coverage_insufficient',
                'method': method_name,
                'mode': mode_label,
                'gate_max_rank': gate_max_rank,
                'gate_min_blur_ratio': gate_min_blur_ratio,
                'gate_coverage': coverage,
                'hint': (
                    'Run the crop-rank and blur backfill scripts over the '
                    'residual pool before enabling the clustering gate.'
                ),
            }

    # asyncio.to_thread — see embedding_reduce.py's matching call site:
    # detect_cluster_backend()'s CUDA probe has been observed to block
    # 30+ seconds on a no-GPU-passthrough container (driver-mismatch
    # error path), long enough to starve the sibling heartbeat coroutine
    # and trigger a false "worker heartbeat stale" job failure.
    backend_info = await asyncio.to_thread(detect_cluster_backend)
    if progress is not None:
        progress.set_backend(
            name=backend_info.name,
            detail=backend_info.detail,
            free_vram_mb=backend_info.free_vram_mb,
        )

    # PIT + sliced parallel fetch by default (~8x faster than scroll
    # on n=347k). Falls back internally to the sequential scroll
    # variant if PIT isn't available on the cluster.
    ids, embeddings = await embedding_reduce.fetch_residual_embeddings_parallel(
        client,
        include_candidate_clusters=recluster_unvalidated,
        candidate_cluster_id_min=RESIDUAL_CLUSTER_ID_OFFSET,
        extra_must=gate_clauses or None,
        progress=progress,
    )
    if len(ids) < MIN_RESIDUALS_FOR_CLUSTERING:
        return {
            'status': 'no_residuals' if not ids else 'too_few_residuals',
            'method': method_name,
            'mode': mode_label,
            'n_residuals': len(ids),
            'n_clusters': 0,
            'n_noise': 0,
            'cluster_id_offset': RESIDUAL_CLUSTER_ID_OFFSET,
            'backend': backend_info.name,
            'backend_detail': backend_info.detail,
        }

    if progress is not None:
        progress.update(processed=0, total=0)

    method_kwargs: dict[str, Any] = {}
    if n_clusters is not None and method_name == 'ivf':
        # Only IVF takes a fixed cluster count; AHC/HDBSCAN derive their own.
        method_kwargs['n_clusters'] = n_clusters
    method = get_method(method_name, **method_kwargs)
    result = await method.fit_predict(
        embeddings,
        backend_info=backend_info,
        progress=progress,
    )

    if progress is not None:
        progress.raise_if_cancelled()

    # Write the method's raw labels as-is (no size-based renumbering) —
    # see the docstring above for why: this must match assign_only_residuals
    # and the ingest-time assign path bit-for-bit so the same crop can't get
    # a different cluster_id depending on which code path clustered it.
    relabeled = np.asarray(result.labels, dtype=np.int64)
    n_clusters = len({int(x) for x in relabeled if int(x) != -1})
    n_noise = int((relabeled == -1).sum())

    # Per-crop centroid distance, aligned 1:1 with ids/labels. Methods
    # without one (IVF's single-bucket fallback, AHC, HDBSCAN) get the
    # member-mean centroid distance so outlier sorts work for every run.
    distances = result.distances
    if distances is not None:
        dist_list = distances.tolist()
    else:
        from src.services.curation.clustering.centroid_distance import member_centroid_distances

        dist_list = member_centroid_distances(embeddings, relabeled)

    # Chunk the bulk write — at production scale (~350k items) a single
    # bulk call exceeds the OpenSearch client's default 30s timeout.
    # Each chunk publishes incremental progress so the dashboard
    # advances during the write phase.
    BULK_CHUNK = 2000
    pairs = list(zip(ids, relabeled.tolist(), dist_list, strict=True))
    if progress is not None:
        progress.update(processed=0, total=len(pairs))
    n_written = 0
    for start in range(0, len(pairs), BULK_CHUNK):
        chunk = pairs[start : start + BULK_CHUNK]
        bulk_body: list[dict[str, Any]] = []
        for crop_id, label, dist in chunk:
            # Residual labels >= 0 get the offset; -1 (noise) stays as
            # -1 so the API renders it as cluster_kind="unassigned".
            new_cid = int(label)
            if new_cid >= 0:
                new_cid += RESIDUAL_CLUSTER_ID_OFFSET
            bulk_body.append({'update': {'_index': items_index(), '_id': crop_id}})
            # Guarded script, not a blind 'doc' update -- clear
            # cluster_subid (stale refine groupings from the doc's previous
            # cluster have no meaning in the new candidate) and set
            # cluster_distance (present for IVF, else null so a stale
            # distance from a prior method can't mislead), but noop if a
            # human relabeled/validated/excluded the doc since it was
            # fetched for this fit.
            bulk_body.append(
                _guarded_class_cluster_write(new_cid, float(dist) if dist is not None else None)
            )
        br = await client.bulk(body=bulk_body, refresh=False)
        if br.get('errors'):
            _log_bulk_write_errors('cluster_residuals', br)
        n_written += len(chunk)
        if progress is not None:
            progress.update(processed=n_written, total=len(pairs))
            progress.raise_if_cancelled()
    if pairs:
        try:
            await client.indices.refresh(index=items_index())
        except Exception as exc:
            logger.debug('curation_cluster_refresh_failed', error=str(exc))

    # Park the gate complement (residual crops too small / blurry to train or
    # assign) and persist the gate policy so ingest + assign_only apply it.
    # When the gate is off, clear any stale policy so the full pool clusters.
    n_parked = 0
    if gate_active:
        n_parked = await _park_gated_residuals(
            client, max_rank=gate_max_rank, min_blur_ratio=gate_min_blur_ratio
        )
    try:
        from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

        ivf_store = IVFCentroidStore()
        ivf_store.save_gate(max_rank=gate_max_rank, min_blur_ratio=gate_min_blur_ratio)
        # Record which pool mode produced the n_trained_on this run just
        # persisted (via ClusterMethod.fit_predict -> IVFCentroidStore.save)
        # so should_retrain_centroids can compare like-for-like later. See
        # its docstring for why this matters.
        if method_name == 'ivf':
            ivf_store.update_metadata(trained_mode=mode_label)
    except Exception as exc:
        logger.warning('curation_cluster_save_gate_failed', error=str(exc))

    summary = {
        'status': 'success',
        'method': result.method,
        'mode': mode_label,
        'n_residuals': len(ids),
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_parked': n_parked,
        'gate_active': gate_active,
        'gate_max_rank': gate_max_rank,
        'gate_min_blur_ratio': gate_min_blur_ratio,
        'cluster_id_offset': RESIDUAL_CLUSTER_ID_OFFSET,
        'backend': backend_info.name,
        'backend_detail': backend_info.detail,
        'cluster_method_backend': result.backend,
        'cluster_method_params': result.params,
        'cluster_method_extra': result.extra,
    }
    record_last_run(summary)
    return summary


async def assign_only_residuals(
    client: AsyncOpenSearch,
    *,
    chunk_size: int = 2000,
    progress: Any = None,
) -> dict[str, Any]:
    """Assign residual crops against the persisted IVF centroids, streaming.

    No retraining: loads the centroids saved by the last
    :func:`cluster_residuals` (IVF) run and streams the residual pool
    one scroll page at a time — assign + bulk-write per page, then drop
    the page from memory. Peak RAM is one page (~8 MB at chunk_size=2000)
    regardless of pool size, so this is the cheap periodic "re-sort the
    residuals with the current centroids" path (vs a full retrain).

    Returns a summary dict. If no centroids are persisted yet, returns
    ``{'status': 'no_centroids'}`` — the caller should run a full
    :func:`cluster_residuals` first.
    """
    from src.services.curation.clustering import embedding_reduce
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

    store = IVFCentroidStore()
    if not store.load():
        return {'status': 'no_centroids', 'method': 'ivf_assign_only'}

    # Apply the gate the last full recluster trained under (if any), so the
    # incremental re-sort matches the centroids' training distribution.
    gate = store.load_gate()
    gate_max_rank = gate.get('max_rank')
    gate_min_blur_ratio = gate.get('min_blur_ratio')
    gate_clauses = gate_must_clauses(gate_max_rank, gate_min_blur_ratio)

    field = embedding_reduce.RESIDUAL_EMBEDDING_FIELD
    filt: list[dict[str, Any]] = [{'exists': {'field': field}}, *gate_clauses]
    must_not: list[dict[str, Any]] = [
        {'term': {'class_validated': True}},
        {'terms': {'class_source': list(embedding_reduce.CONFIDENT_CLASS_SOURCES)}},
        {'term': {'class_excluded': True}},
    ]
    query = {'bool': {'filter': filt, 'must_not': must_not}}

    total_estimate = 0
    if progress is not None:
        try:
            cnt = await client.count(index=items_index(), body={'query': query})
            total_estimate = int(cnt.get('count', 0))
            progress.update(processed=0, total=total_estimate)
        except Exception as exc:
            logger.debug('curation_ivf_assign_count_failed', error=str(exc))

    body: dict[str, Any] = {'size': chunk_size, '_source': [field], 'query': query}
    resp = await client.search(index=items_index(), body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    n_written = 0
    try:
        while True:
            hits = resp['hits']['hits']
            if not hits:
                break
            chunk_ids: list[str] = []
            chunk_embs: list[np.ndarray] = []
            for h in hits:
                emb = (h.get('_source') or {}).get(field)
                if emb is None:
                    continue
                chunk_ids.append(h['_id'])
                chunk_embs.append(np.asarray(emb, dtype=np.float32))
            if chunk_embs:
                matrix = np.vstack(chunk_embs)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = matrix / np.where(norms == 0, 1.0, norms)
                labels, dists = store.assign_batch_with_distances(matrix)
                bulk_body: list[dict[str, Any]] = []
                for crop_id, label, dist in zip(
                    chunk_ids, labels.tolist(), dists.tolist(), strict=True
                ):
                    new_cid = int(label) + RESIDUAL_CLUSTER_ID_OFFSET
                    bulk_body.append({'update': {'_index': items_index(), '_id': crop_id}})
                    # Guarded script — see cluster_residuals above.
                    bulk_body.append(_guarded_class_cluster_write(new_cid, float(dist)))
                br = await client.bulk(body=bulk_body, refresh=False)
                if br.get('errors'):
                    _log_bulk_write_errors('assign_only_residuals', br)
                n_written += len(chunk_ids)
                if progress is not None:
                    progress.update(processed=n_written, total=total_estimate or n_written)
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
                logger.debug('curation_ivf_assign_clear_scroll_failed', error=str(exc))

    if n_written:
        try:
            await client.indices.refresh(index=items_index())
        except Exception as exc:
            logger.debug('curation_ivf_assign_refresh_failed', error=str(exc))

    # Park the gate complement so newly-gated crops don't linger in stale
    # candidate buckets after an incremental re-sort.
    n_parked = 0
    if gate_clauses:
        n_parked = await _park_gated_residuals(
            client, max_rank=gate_max_rank, min_blur_ratio=gate_min_blur_ratio
        )

    summary = {
        'status': 'success',
        'method': 'ivf_assign_only',
        'n_assigned': n_written,
        'n_parked': n_parked,
        'gate_active': bool(gate_clauses),
        'n_clusters': store.n_clusters,
        'cluster_id_offset': RESIDUAL_CLUSTER_ID_OFFSET,
        'centroids_trained_at': store.metadata.get('trained_at'),
    }
    record_last_run(summary)
    return summary
