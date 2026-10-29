"""Cost of average-linkage clustering inside one detector class at the size of a real project.

Dense: scipy average linkage on the full cosine-distance matrix of one key (memory grows with
the square of its size). Sparse: average linkage restricted to the edges of a mutual k-nearest-
neighbour graph (scikit-learn's connectivity-constrained agglomeration), whose cost grows with
the number of edges. Both are cut at one cosine-distance height.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering

from scripts.bench.cluster_eval_metrics import UNASSIGNED


if TYPE_CHECKING:
    from numpy.typing import NDArray

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]

NEIGHBOURS = 30
CHUNK = 2048


def condensed_cosine_distance(vectors: Floats, chunk: int = CHUNK) -> Floats:
    """The condensed (upper triangle, row-major) 1 - cosine of unit rows, in float64, floored at 0."""
    n = len(vectors)
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    start = 0
    for i0 in range(0, n, chunk):
        sims = vectors[i0 : i0 + chunk] @ vectors.T
        for r in range(sims.shape[0]):
            i = i0 + r
            width = n - i - 1
            out[start : start + width] = np.clip(1.0 - sims[r, i + 1 :], 0.0, None)
            start += width
    return out


def dense_average_linkage(vectors: Floats) -> Floats:
    """The average-linkage dendrogram of one key, from the full distance matrix."""
    return np.asarray(linkage(condensed_cosine_distance(vectors), method='average'))


def cut_dendrogram(z: Floats, cut: float) -> Ints:
    """Cluster codes of a dendrogram cut at one height."""
    return np.asarray(fcluster(z, cut, 'distance'), dtype=np.int64) - 1


def dense_average_partition(vectors: Floats, cut: float) -> Ints:
    """Average-linkage clusters of one key at ``cut``, from the full distance matrix."""
    if len(vectors) == 1:
        return np.zeros(1, dtype=np.int64)
    return cut_dendrogram(dense_average_linkage(vectors), cut)


def cluster_shape(part: Ints) -> dict[str, float | int]:
    """Number of clusters and the share of the key that sits in the largest one."""
    sizes = np.bincount(part)
    return {'clusters': len(sizes), 'largest_cluster_share': float(sizes.max() / len(part))}


def mutual_knn_graph(vectors: Floats, k: int = NEIGHBOURS, chunk: int = CHUNK) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency of the pairs that list each other among their k nearest."""
    n = len(vectors)
    k = min(k, n - 1)
    rows, cols = [], []
    for i0 in range(0, n, chunk):
        sims = vectors[i0 : i0 + chunk] @ vectors.T
        sims[np.arange(sims.shape[0]), np.arange(i0, i0 + sims.shape[0])] = -np.inf
        nearest = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows.append(np.repeat(np.arange(i0, i0 + sims.shape[0]), k))
        cols.append(nearest.ravel())
    adj = sp.csr_matrix(
        (
            np.ones(sum(len(r) for r in rows), dtype=np.int8),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n, n),
    )
    return sp.csr_matrix(adj.minimum(adj.T))


def sparse_average_partition(vectors: Floats, cut: float, k: int = NEIGHBOURS) -> Ints:
    """Average-linkage clusters of one key at ``cut`` over a mutual k-NN graph only."""
    if len(vectors) < 3:
        return dense_average_partition(vectors, cut)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=cut,
        metric='cosine',
        linkage='average',
        connectivity=mutual_knn_graph(vectors, k),
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # a disconnected graph is completed by scikit-learn
        return np.asarray(model.fit_predict(vectors), dtype=np.int64)


def within_key(vectors: Floats, keys: Any, cut: float, partition_one: Any) -> Ints:
    """Apply a one-key partition function to every key; codes are unique across keys."""
    k = np.asarray(keys).astype(str)
    labels = np.full(len(vectors), UNASSIGNED, dtype=np.int64)
    offset = 0
    for name in sorted(set(k.tolist())):
        idx = np.nonzero(k == name)[0]
        part = partition_one(vectors[idx], cut)
        labels[idx] = part + offset
        offset += int(part.max()) + 1
    return labels
