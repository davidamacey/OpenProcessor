"""FAISS IVF clustering for the residual pool.

Pivoted to as the default residual-pool clusterer on 2026-05-23 after
HDBSCAN's density-based stability selection collapsed on the
continuously-dense vehicle-embedding manifold (parameter sweep over
``min_cluster_size`` ∈ {5, 20, 50, 100, 200} and
``cluster_selection_method`` ∈ {eom, leaf} produced either one
mega-cluster or 100 % noise — there are no density gaps for HDBSCAN to
exploit).

IVF (Inverted File) is a *partitioning* method, not a density method:
it just runs k-means to find ``n_clusters`` centroids, then assigns
every embedding to its nearest centroid. That's a much better fit for
the operational need ("give the human a manageable number of buckets
to browse") because:

* Cluster count is fixed (``n_clusters=512``), so no tuning crisis.
* Every embedding gets a cluster — no ``-1`` noise.
* Bucket sizes are roughly balanced by the k-means update rule.
* Same crop lands in the same bucket across reruns (until the next
  retrain), so a human mid-label session isn't disrupted.

Limitations (documented so we don't re-litigate them later):

* IVF doesn't *discover* meaningful boundaries — a Voronoi cell can
  split a tight visual group across two centroids if they happen to
  fall near the boundary.
* Cluster IDs are not interpretable: bucket #173 is "things near
  centroid 173", not "sportbikes".
* Retraining changes the centroid positions, so the cluster_id of a
  given crop can shift. We re-run only when the residual pool has
  grown substantially.

Knobs match :py:data:`src.services.clustering.DEFAULT_CONFIGS[ClusterIndex.VEHICLES]`
(n_clusters=512, niter=50, nredo=5, vectors_per_cluster=6) so this
method produces results comparable to the existing `vehicles` FAISS
index that the rest of the system already trusts. This is a
deliberate constant match, not a shared import — the two clustering
subsystems (this residual-pool clusterer and the unrelated
``src/services/clustering.py`` visual-search module, see plan §0.11)
stay independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from src.core.logging import get_logger
from src.services.curation.clustering.methods.base import ClusterResult


if TYPE_CHECKING:
    from src.services.curation.clustering.backend import BackendInfo


logger = get_logger(__name__)


# Matches DEFAULT_CONFIGS[ClusterIndex.VEHICLES] in src/services/clustering.py.
IVF_DEFAULT_N_CLUSTERS = 512
IVF_DEFAULT_NITER = 50
IVF_DEFAULT_NREDO = 5
IVF_MIN_VECTORS_PER_CLUSTER = 6
# Below this many residuals we can't fit even one usable centroid per
# minimum bucket; skip training and fall through to a single bucket.
IVF_MIN_TRAIN_VECTORS = IVF_MIN_VECTORS_PER_CLUSTER * 32

# Train k-means on at most this many vectors. Above it, a random sample
# is drawn — k-means centroids converge on a representative sample at
# n>>k (FAISS docs recommend ~256*k training points; 50k >> 256*512 is
# already generous), and sampling keeps the training transient bounded
# (~200 MB for 50k x 1024 f32) regardless of how large the pool grows.
# The FULL pool is still assigned against the fitted centroids.
IVF_MAX_TRAIN_SAMPLE = 50_000


class IVFMethod:
    """FAISS IVF (k-means partitioning) on the residual embedding pool.

    Trains on a bounded random sample (Option A) and persists the fitted
    centroids via :class:`IVFCentroidStore` so the ingest path and
    periodic reassign can reuse them without retraining.
    """

    name: ClassVar[str] = 'ivf'

    def __init__(
        self,
        *,
        n_clusters: int = IVF_DEFAULT_N_CLUSTERS,
        niter: int = IVF_DEFAULT_NITER,
        nredo: int = IVF_DEFAULT_NREDO,
        max_train_sample: int = IVF_MAX_TRAIN_SAMPLE,
        persist: bool = True,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.niter = int(niter)
        self.nredo = int(nredo)
        self.max_train_sample = int(max_train_sample)
        self.persist = persist

    async def fit_predict(
        self,
        embeddings: np.ndarray,
        *,
        backend_info: BackendInfo,
        progress: Any = None,
    ) -> ClusterResult:
        import asyncio as _asyncio

        if progress is not None:
            progress.raise_if_cancelled()

        n, d = embeddings.shape
        # If the pool is too small to give every centroid at least
        # IVF_MIN_VECTORS_PER_CLUSTER, fall back to a single cluster
        # rather than training an under-fit k-means model.
        n_clusters = min(self.n_clusters, max(1, n // IVF_MIN_VECTORS_PER_CLUSTER))

        if n < IVF_MIN_TRAIN_VECTORS:
            logger.info(
                'kb_ivf_too_few_vectors_single_bucket', n=n, threshold=IVF_MIN_TRAIN_VECTORS
            )
            labels = np.zeros(n, dtype=np.int64)
            return ClusterResult(
                labels=labels,
                method=self.name,
                backend='faiss_cpu',
                params=self._params(actual_n_clusters=1, dim=d),
                extra={
                    'n_clusters': 1,
                    'n_noise': 0,
                    'noise_pct': 0.0,
                    'largest_cluster_size': int(n),
                    'largest_cluster_pct': 1.0,
                    'reason': 'too_few_vectors_for_training',
                },
            )

        prefer_gpu = backend_info.name == 'gpu'
        labels, distances, used_backend, extra = await _asyncio.to_thread(
            self._fit, embeddings, n_clusters, prefer_gpu
        )
        logger.info(
            'kb_ivf_done',
            n=n,
            n_clusters=extra.get('n_clusters'),
            backend=used_backend,
            largest_pct=extra.get('largest_cluster_pct'),
        )
        return ClusterResult(
            labels=labels,
            distances=distances,
            method=self.name,
            backend=used_backend,
            params=self._params(actual_n_clusters=n_clusters, dim=d),
            extra=extra,
        )

    def _params(self, *, actual_n_clusters: int, dim: int) -> dict[str, Any]:
        return {
            'n_clusters_requested': self.n_clusters,
            'n_clusters_actual': actual_n_clusters,
            'niter': self.niter,
            'nredo': self.nredo,
            'embedding_dim': dim,
            'max_train_sample': self.max_train_sample,
        }

    def _fit(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        prefer_gpu: bool,
    ) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
        from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

        x = np.ascontiguousarray(embeddings, dtype=np.float32)
        n = x.shape[0]

        # Sample-train (Option A): k-means on a bounded random subset so
        # the training transient stays flat as the pool grows. The full
        # pool is still assigned against the fitted centroids below.
        if n > self.max_train_sample:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=self.max_train_sample, replace=False)
            train_x = np.ascontiguousarray(x[idx], dtype=np.float32)
        else:
            train_x = x

        try:
            import faiss

            use_gpu = prefer_gpu and faiss.get_num_gpus() > 0
        except Exception:
            use_gpu = False

        store = IVFCentroidStore()
        centroids, final_obj = store.train(
            train_x,
            n_clusters=n_clusters,
            niter=self.niter,
            nredo=self.nredo,
            use_gpu=use_gpu,
        )

        # Persist centroids so the ingest path + periodic reassign reuse
        # them without retraining. Side-effect of every training run.
        if self.persist:
            try:
                store.save(
                    centroids,
                    metadata={
                        'sample_size': len(train_x),
                        'n_trained_on': int(n),
                        'niter': self.niter,
                        'nredo': self.nredo,
                        'kmeans_obj': final_obj,
                        'backend': 'faiss_gpu' if use_gpu else 'faiss_cpu',
                    },
                )
            except Exception as exc:
                logger.warning('kb_ivf_persist_failed', error=str(exc))

        labels, distances = store.assign_batch_with_distances(x)
        backend = 'faiss_gpu' if use_gpu else 'faiss_cpu'
        extra = _summarize_labels(labels)
        extra['final_objective'] = final_obj
        extra['train_sample_size'] = len(train_x)
        extra['centroids_persisted'] = bool(self.persist)
        extra['mean_cluster_distance'] = float(distances.mean()) if len(distances) else 0.0
        return labels, distances, backend, extra


def _summarize_labels(labels: np.ndarray) -> dict[str, Any]:
    """Stats for the dashboard. IVF emits no -1; n_noise always 0."""
    if len(labels) == 0:
        return {
            'n_clusters': 0,
            'n_noise': 0,
            'noise_pct': 0.0,
            'largest_cluster_size': 0,
            'largest_cluster_pct': 0.0,
        }
    _, counts = np.unique(labels, return_counts=True)
    n_clusters = len(counts)
    largest = int(counts.max())
    return {
        'n_clusters': n_clusters,
        'n_noise': 0,
        'noise_pct': 0.0,
        'largest_cluster_size': largest,
        'largest_cluster_pct': largest / len(labels),
    }


__all__ = [
    'IVF_DEFAULT_NITER',
    'IVF_DEFAULT_NREDO',
    'IVF_DEFAULT_N_CLUSTERS',
    'IVFMethod',
]
