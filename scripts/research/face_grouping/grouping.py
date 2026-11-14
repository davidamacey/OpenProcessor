"""Grouping methods over L2-normalised embeddings; each returns one label per row, -1 = unassigned.

Documentation consulted: scipy.cluster.hierarchy.linkage / fcluster (average and complete linkage,
cosine distance), faiss IndexFlatIP exact search and range_search, scipy.sparse.csgraph.connected_components,
the ``chinese-whispers`` 0.9.0 package (``chinese_whispers(graph, weighting, iterations, seed)``) and
``finch-clust`` 0.2.3 (``FINCH(data, distance='cosine', ann_threshold=...)``). The Immich rule is the one in
``person.service.ts`` ``handleRecognizeFaces`` (see README).
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist


UNASSIGNED = -1


@dataclass(frozen=True)
class Graph:
    """Exact neighbours of every row (self excluded, best first) and every pair above ``floor``."""

    x: np.ndarray
    nn_idx: np.ndarray
    nn_sim: np.ndarray
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_sim: np.ndarray
    floor: float

    @property
    def n(self) -> int:
        return len(self.x)


def build_graph(x: np.ndarray, k: int, floor: float) -> Graph:
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(x)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    kk = min(k + 1, n)
    sim, idx = index.search(x, kk)
    sim = np.where(idx == np.arange(n)[:, None], -np.inf, sim)  # drop self, even among duplicates
    order = np.argsort(-sim, axis=1, kind='stable')[:, : kk - 1]
    nn_sim = np.take_along_axis(sim, order, axis=1)
    nn_idx = np.take_along_axis(idx, order, axis=1)
    lims, dist, nbr = index.range_search(x, float(floor))
    rows = np.repeat(np.arange(n), np.diff(lims).astype(np.int64))
    nbr = nbr.astype(np.int64)
    keep = rows < nbr
    return Graph(x, nn_idx, nn_sim, rows[keep], nbr[keep], dist[keep], float(floor))


def _components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    adj = sp.coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    labels: np.ndarray = connected_components(adj, directed=False)[1]
    return labels.astype(np.int64)


def single_linkage(g: Graph, sim: float) -> np.ndarray:
    """Connected components of every pair at cosine ``sim`` or above (today's grouping)."""
    if sim < g.floor:
        raise ValueError(f'threshold {sim} is below the graph floor {g.floor}')
    keep = g.edge_sim >= sim
    return _components(g.n, g.edge_i[keep], g.edge_j[keep])


def linkage_tree(x: np.ndarray, method: str) -> np.ndarray:
    """Exact agglomerative tree over cosine distance; cut it at many thresholds with ``cut_tree``."""
    tree: np.ndarray = linkage(pdist(np.asarray(x, dtype=np.float64), 'cosine'), method=method)
    return tree


def cut_tree(tree: np.ndarray, sim: float) -> np.ndarray:
    labels: np.ndarray = fcluster(tree, t=1.0 - sim, criterion='distance')
    return labels.astype(np.int64)


def agglomerative(x: np.ndarray, method: str, sim: float) -> np.ndarray:
    return cut_tree(linkage_tree(x, method), sim)


def immich_core(g: Graph, max_distance: float, min_faces: int) -> np.ndarray:
    """Immich's greedy recognition: a face with ``min_faces`` hits within ``max_distance`` founds a group.

    Hits count the face itself (the SQL search returns it), as Immich does. A face joins the group of its
    first hit that has one, else of its nearest grouped neighbour within ``max_distance``; a face with no
    group is deferred once and tried again after the first pass; only a core face creates a group.
    """
    n, floor_sim = g.n, 1.0 - max_distance
    label = np.full(n, UNASSIGNED, dtype=np.int64)
    next_group = 0
    near = g.nn_sim >= floor_sim
    hits = 1 + near[:, : max(min_faces - 1, 0)].sum(axis=1)

    def place(i: int) -> bool:
        nonlocal next_group
        if min_faces > 1 and hits[i] == 1:
            return False
        first = [
            int(j) for j in g.nn_idx[i, : max(min_faces - 1, 0)][near[i, : max(min_faces - 1, 0)]]
        ]
        group = next((label[j] for j in first if label[j] != UNASSIGNED), UNASSIGNED)
        if group == UNASSIGNED:
            group = next(
                (
                    label[j]
                    for j, ok in zip(g.nn_idx[i], near[i], strict=True)
                    if ok and label[j] != UNASSIGNED
                ),
                UNASSIGNED,
            )
        if group == UNASSIGNED and hits[i] >= min_faces:
            group, next_group = next_group, next_group + 1
        if group == UNASSIGNED:
            return False
        label[i] = group
        return True

    deferred = [i for i in range(n) if not place(i)]
    for i in deferred:
        place(i)
    return label


def _merge_rounds(
    x: np.ndarray, label: np.ndarray, merge_sim: float, max_rounds: int = 60
) -> np.ndarray:
    """Merge mutually-best cluster pairs whose mean cross similarity is at least ``merge_sim``, repeat."""
    label = np.unique(label, return_inverse=True)[1].astype(np.int64)
    x32 = x.astype(np.float32)
    for _ in range(max_rounds):
        c = int(label.max()) + 1
        ind = sp.csr_matrix(
            (np.ones(len(label), dtype=np.float32), (label, np.arange(len(label)))),
            shape=(c, len(label)),
        )
        sums = np.asarray(ind @ x32)
        size = np.bincount(label, minlength=c).astype(np.float32)
        avg = (sums @ sums.T) / np.outer(size, size)  # mean pairwise cosine between two clusters
        np.fill_diagonal(avg, -np.inf)
        best = avg.argmax(axis=1)
        top = avg[np.arange(c), best]
        a = np.arange(c)
        mutual = (best[best] == a) & (top >= merge_sim) & (a < best)
        if not mutual.any():
            break
        remap = a.copy()
        remap[best[mutual]] = a[mutual]
        label = np.unique(remap[label], return_inverse=True)[1].astype(np.int64)
    return label


def mutual_knn(g: Graph, k: int, floor: float, merge_sim: float | None) -> np.ndarray:
    """Components of the mutual k-nearest-neighbour graph above an edge floor, then a cluster-level merge."""
    kk = min(k, g.nn_idx.shape[1])
    rows = np.repeat(np.arange(g.n), kk)
    cols = g.nn_idx[:, :kk].ravel()
    ok = g.nn_sim[:, :kk].ravel() >= floor
    knn = sp.csr_matrix((np.ones(ok.sum(), dtype=np.int8), (rows[ok], cols[ok])), shape=(g.n, g.n))
    mutual = knn.multiply(knn.T).tocoo()
    label = _components(g.n, mutual.row, mutual.col)
    return label if merge_sim is None else _merge_rounds(g.x, label, merge_sim)


def chinese_whisper_groups(g: Graph, floor: float, seed: int, iterations: int = 20) -> np.ndarray:
    """Chinese Whispers on the kNN graph, edges weighted by cosine and cut at ``floor``."""
    import networkx as nx
    from chinese_whispers import chinese_whispers

    graph = nx.Graph()
    graph.add_nodes_from(range(g.n))
    ok = g.nn_sim >= floor
    rows = np.repeat(np.arange(g.n), g.nn_idx.shape[1]).reshape(g.nn_idx.shape)
    graph.add_weighted_edges_from(
        zip(rows[ok].tolist(), g.nn_idx[ok].tolist(), g.nn_sim[ok].tolist(), strict=True)
    )
    chinese_whispers(graph, weighting='top', iterations=iterations, seed=seed)
    raw = [graph.nodes[i]['label'] for i in range(g.n)]
    return np.unique(np.asarray(raw), return_inverse=True)[1].astype(np.int64)


def finch_levels(x: np.ndarray) -> list[np.ndarray]:
    """One partition per FINCH hierarchy level (FINCH has no threshold; each level is a point)."""
    from finch import FINCH

    parts = FINCH(
        np.asarray(x, dtype=np.float32),
        distance='cosine',
        ann_threshold=len(x) + 1,
        ensure_early_exit=True,
        verbose=False,
        random_state=0,
    )[0]
    return [parts[:, level].astype(np.int64) for level in range(parts.shape[1])]
