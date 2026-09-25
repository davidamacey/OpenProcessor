"""AHC complete-linkage cosine on a sparse kNN connectivity graph.

Lifted from :py:mod:`src.services.curation.clustering.orchestrator` on 2026-05-22
when the clustering layer was abstracted behind
:class:`src.services.curation.clustering.methods.base.ClusterMethod`.

Behavior is unchanged. Knobs:

* ``linkage = "complete"`` — strongest bleed-over guard.
* ``metric  = "cosine"``   — matches the embedding training objective.
* ``distance_threshold = 0.25`` — ~75 % cosine similarity; shared with
  the refine endpoint so both surfaces are calibrated identically.
* ``k = 30`` — sparse kNN connectivity matrix; keeps memory O(n·k)
  instead of O(n²). cuML on GPU builds the kNN, sklearn on CPU is the
  fallback.

Known limitation (the reason IVF is the default now, not AHC): sklearn's
hierarchical merge loop is a C call that doesn't fully release the GIL,
so on large n (~300k+) the worker's heartbeat task can starve and the
watchdog declares the worker dead. Use ``clustering_method="ahc"`` only
for smaller pools or when the watchdog threshold has been bumped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from src.core.logging import get_logger
from src.services.curation.clustering.methods.base import ClusterResult


if TYPE_CHECKING:
    from src.services.curation.clustering.backend import BackendInfo


logger = get_logger(__name__)


AHC_DISTANCE_THRESHOLD = 0.25
AHC_LINKAGE = 'complete'
AHC_METRIC = 'cosine'
KNN_NEIGHBOURS = 30


async def _build_knn_graph(
    embeddings: np.ndarray,
    k: int,
    backend_info: BackendInfo,
) -> tuple[Any, str]:
    """Symmetric sparse cosine kNN adjacency, GPU-preferred."""
    import asyncio as _asyncio

    def _gpu_knn() -> Any:
        from cuml.neighbors import NearestNeighbors as cuNN  # type: ignore[import-not-found]

        # CM-8: cuML's kneighbors_graph includes each point as its own
        # nearest neighbor (n_neighbors=k+1 asks for k *other* points
        # plus self); sklearn's CPU path below passes include_self=False
        # instead. Left in, the self-loop is an always-1.0-similarity
        # edge on every node's row, silently thickening the GPU graph's
        # connectivity relative to the CPU graph for the same k -- drop
        # the diagonal so both backends produce the same connectivity
        # shape for the same embeddings.
        nn = cuNN(n_neighbors=k + 1, metric='cosine')
        nn.fit(embeddings)
        graph = nn.kneighbors_graph(embeddings, mode='connectivity')
        if hasattr(graph, 'get'):
            graph = graph.get()
        try:
            import scipy.sparse as sp

            graph = sp.csr_matrix(graph)
            graph.setdiag(0)
            graph.eliminate_zeros()
            return 0.5 * (graph + graph.T)
        except Exception:
            return graph

    def _cpu_knn() -> Any:
        from sklearn.neighbors import kneighbors_graph as skg

        g = skg(
            embeddings,
            n_neighbors=k,
            metric='cosine',
            mode='connectivity',
            include_self=False,
        )
        return 0.5 * (g + g.T)

    if backend_info.name == 'gpu':
        try:
            graph = await _asyncio.to_thread(_gpu_knn)
            return graph, 'cuml'
        except Exception as exc:
            logger.warning('curation_ahc_gpu_knn_failed_fallback_cpu', error=str(exc))
    graph = await _asyncio.to_thread(_cpu_knn)
    return graph, 'sklearn'


class AHCMethod:
    """Agglomerative complete-linkage cosine on a sparse kNN graph."""

    name: ClassVar[str] = 'ahc'

    def __init__(
        self,
        *,
        distance_threshold: float = AHC_DISTANCE_THRESHOLD,
        linkage: str = AHC_LINKAGE,
        metric: str = AHC_METRIC,
        k: int = KNN_NEIGHBOURS,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.metric = metric
        self.k = k

    async def fit_predict(
        self,
        embeddings: np.ndarray,
        *,
        backend_info: BackendInfo,
        progress: Any = None,
    ) -> ClusterResult:
        import asyncio as _asyncio

        from sklearn.cluster import AgglomerativeClustering

        k = min(self.k, len(embeddings) - 1)
        connectivity, knn_backend = await _build_knn_graph(embeddings, k, backend_info)
        logger.info('curation_ahc_knn_graph_built', backend=knn_backend, k=k, n=len(embeddings))

        if progress is not None:
            progress.raise_if_cancelled()

        def _fit() -> np.ndarray:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                linkage=self.linkage,
                metric=self.metric,
                connectivity=connectivity,
            )
            return clusterer.fit_predict(embeddings).astype(np.int64)

        labels = await _asyncio.to_thread(_fit)
        return ClusterResult(
            labels=labels,
            method=self.name,
            backend=knn_backend,
            params={
                'distance_threshold': self.distance_threshold,
                'linkage': self.linkage,
                'metric': self.metric,
                'k': k,
                'connectivity': 'knn',
            },
            extra={},
        )


__all__ = [
    'AHC_DISTANCE_THRESHOLD',
    'AHC_LINKAGE',
    'AHC_METRIC',
    'KNN_NEIGHBOURS',
    'AHCMethod',
]
