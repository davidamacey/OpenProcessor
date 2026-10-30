"""Baseline partitions for the Leiden prototype: an emulation of the current IVF + AHC pipeline,
HDBSCAN (leaf) and FINCH. Research code, CPU. Heavy imports are lazy.

The IVF emulation follows the refresh worker's rule (K = min(512, n // 6) FAISS k-means centroids,
trained on at most 50k rows, every item assigned to its nearest centroid) and the AHC refine route
(complete linkage, cosine, distance threshold 0.25, buckets of 4 to 8000 members). The product
refines on demand; here every bucket is refined, which is the most favourable reading of it.
"""

from __future__ import annotations

import numpy as np


AHC_THRESHOLD = 0.25
AHC_MIN, AHC_MAX = 4, 8000


def ivf_buckets(vectors: np.ndarray, *, seed: int = 0, threads: int = 6) -> np.ndarray:
    import faiss

    faiss.omp_set_num_threads(threads)
    n, dim = vectors.shape
    k = min(512, max(2, n // 6))
    rng = np.random.default_rng(seed)
    train = vectors[rng.choice(n, size=min(n, 50_000), replace=False)]
    km = faiss.Kmeans(dim, k, niter=20, seed=seed, spherical=True)
    km.train(train)
    return km.index.search(vectors, 1)[1][:, 0].astype(np.int64)


def ahc_refine(vectors: np.ndarray, bucket: np.ndarray) -> np.ndarray:
    """Complete-linkage cosine AHC inside each IVF bucket; returns a global cluster code."""
    from sklearn.cluster import AgglomerativeClustering

    out = np.zeros(len(bucket), dtype=np.int64)
    nxt = 0
    for b in np.unique(bucket):
        members = np.flatnonzero(bucket == b)
        if AHC_MIN <= len(members) <= AHC_MAX:
            sub = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=AHC_THRESHOLD,
                metric='cosine',
                linkage='complete',
            ).fit_predict(vectors[members])
        else:
            sub = np.zeros(len(members), dtype=np.int64)
        out[members] = nxt + sub
        nxt += int(sub.max()) + 1
    return out


def hdbscan_leaf(vectors: np.ndarray, *, min_cluster_size: int = 5, threads: int = 6) -> np.ndarray:
    import hdbscan

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size,
        cluster_selection_method='leaf',
        metric='euclidean',  # on unit vectors, a monotone function of cosine
        core_dist_n_jobs=threads,
    )
    return np.asarray(model.fit_predict(vectors)).astype(np.int64)


def finch_partitions(vectors: np.ndarray) -> list[np.ndarray]:
    """FINCH hierarchy, finest first (first-neighbour clustering, cosine)."""
    from finch import FINCH

    c, _, _ = FINCH(vectors, distance='cosine', verbose=False)
    return [c[:, i].astype(np.int64) for i in range(c.shape[1])]
