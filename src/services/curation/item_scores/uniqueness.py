"""Uniqueness scorer — k-NN density/typicality.

FiftyOne's ``compute_uniqueness``: local k-NN density, distinct from
"representativeness" (global proximity to the assigned cluster centroid,
already available today as ``cluster_distance``, zero new math).
A crop with few near neighbours in embedding space is locally novel even if
it sits close to its cluster's centroid.

Implementation reuses the already-persisted 512 IVF centroids
(:class:`src.services.curation.clustering.methods.ivf_store.IVFCentroidStore`)
as the coarse quantizer for a ``faiss.IndexIVFFlat`` — this is the validated
production partition, not a new geometry. ``nprobe`` buckets are
searched per query instead of a brute-force flat kNN, which is the
difference between ~2-8 min and ~30-60 min CPU at 350k crops.

Score = mean cosine distance to the ``k`` nearest neighbours, min-max
normalized to ``[0, 1]`` across the batch. Higher = more unique (locally
sparse neighbourhood); lower = redundant (near-duplicate cluster core).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from src.services.curation.item_scores.base import ScoreResult


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


UNIQUENESS_VERSION = 'v1'
DEFAULT_KNN_K = 16
DEFAULT_NPROBE = 12


def _knn_k() -> int:
    try:
        return max(1, int(os.environ.get('OP_SCORES_KNN_K', str(DEFAULT_KNN_K))))
    except ValueError:
        return DEFAULT_KNN_K


def _nprobe() -> int:
    try:
        return max(1, int(os.environ.get('OP_SCORES_NPROBE', str(DEFAULT_NPROBE))))
    except ValueError:
        return DEFAULT_NPROBE


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def compute_uniqueness(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    *,
    k: int = DEFAULT_KNN_K,
    nprobe: int = DEFAULT_NPROBE,
) -> np.ndarray:
    """Pure math core — no OpenSearch, no persisted store. Testable directly.

    Args:
        embeddings: ``(n, d)`` float32, unit-norm (dot == cosine).
        centroids: ``(nlist, d)`` float32, unit-norm — reused as the
            IVFFlat coarse quantizer (``nlist`` = number of IVF buckets).
        k: neighbours per query (excluding self).
        nprobe: IVF buckets probed per query.

    Returns:
        ``(n,)`` float32 array in ``[0, 1]``, min-max normalized across the
        batch. ``n < 2`` returns all-zeros (no neighbours to compare).
    """
    import faiss

    n = embeddings.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float32)

    x = np.ascontiguousarray(embeddings, dtype=np.float32)
    c = np.ascontiguousarray(centroids, dtype=np.float32)
    dim = x.shape[1]
    nlist = c.shape[0]

    quantizer = faiss.IndexFlatL2(dim)
    quantizer.add(c)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    # Reusing externally-trained centroids as the coarse quantizer: FAISS's
    # own .train() only fits the quantizer, which we've already supplied.
    # Marking is_trained=True lets .add() proceed without re-running k-means.
    index.is_trained = True
    index.nprobe = max(1, min(nprobe, nlist))
    index.add(x)

    # +1 because the query point itself is always its own nearest neighbour
    # (distance 0) when it's already in the index.
    kk = min(k + 1, n)
    sq_l2, idx = index.search(x, kk)

    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row_idx = idx[i]
        row_d = sq_l2[i]
        neighbor_d = [float(row_d[j]) for j in range(kk) if row_idx[j] != i and row_idx[j] != -1]
        if not neighbor_d:
            # Degenerate: every returned hit was self (tiny nlist/nprobe in
            # tests) — fall back to including them rather than scoring 0.
            neighbor_d = [float(row_d[j]) for j in range(kk) if row_idx[j] != -1]
        # cosine_dist = sq_l2 / 2 for unit-norm vectors (matches ivf_store's
        # assign_batch_with_distances convention).
        scores[i] = float(np.mean(neighbor_d)) / 2.0 if neighbor_d else 0.0

    lo, hi = float(scores.min()), float(scores.max())
    if hi > lo:
        return ((scores - lo) / (hi - lo)).astype(np.float32)
    return np.zeros(n, dtype=np.float32)


class UniquenessScorer:
    """``crop_scores`` registry entry wrapping :func:`compute_uniqueness`."""

    name: ClassVar[str] = 'uniqueness'
    writes: ClassVar[tuple[str, ...]] = (
        'uniqueness_score',
        'uniqueness_method',
        'uniqueness_version',
        'uniqueness_scored_at',
    )
    version: ClassVar[str] = UNIQUENESS_VERSION
    pool: ClassVar[str] = 'residual'

    def __init__(self, centroids: np.ndarray | None = None) -> None:
        """``centroids`` overrides the persisted IVF store — used by tests
        to inject synthetic centroids without touching disk."""
        self._centroids_override = centroids

    def _load_centroids(self) -> np.ndarray:
        if self._centroids_override is not None:
            return self._centroids_override
        from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

        store = IVFCentroidStore()
        if not store.load():
            raise RuntimeError(
                'uniqueness scorer requires a trained IVF centroid store '
                '(none found — run the clustering pipeline at least once first)'
            )
        # reconstruct_n pulls the K stored centroid vectors back out of the
        # flat index so we can hand them to a fresh IVFFlat as coarse quantizer.
        index = store._index
        return np.asarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)

    async def score(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        *,
        opensearch: AsyncOpenSearch | None = None,  # noqa: ARG002 - protocol uniformity
        progress: Any = None,  # noqa: ARG002 - protocol uniformity
    ) -> ScoreResult:
        centroids = self._load_centroids()
        scores = compute_uniqueness(embeddings, centroids, k=_knn_k(), nprobe=_nprobe())
        now = _now_iso()
        fields: dict[str, dict[str, Any]] = {
            crop_id: {
                'uniqueness_score': float(scores[i]),
                'uniqueness_method': 'ivf_knn_cosine',
                'uniqueness_version': self.version,
                'uniqueness_scored_at': now,
            }
            for i, crop_id in enumerate(ids)
        }
        return ScoreResult(
            scorer=self.name,
            version=self.version,
            scored_at=now,
            fields=fields,
            n_scored=len(fields),
            extra={'k': _knn_k(), 'nprobe': _nprobe()},
        )


__all__ = ['UNIQUENESS_VERSION', 'UniquenessScorer', 'compute_uniqueness']
