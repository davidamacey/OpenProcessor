"""Graph construction and partition refinement for the Leiden prototype (research code).

Everything here works on L2-normalised vectors and returns plain numpy / scipy objects so the CPU
and the GPU runners share the construction rules. Heavy imports (faiss, igraph, leidenalg) are
lazy: the GPU environment does not carry igraph.
"""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np
import scipy.sparse as sp


UNASSIGNED = -1


def knn_cpu(
    vectors: np.ndarray, k: int, *, exact_below: int = 150_000, threads: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine kNN (self excluded) with FAISS on the CPU: exact flat below ``exact_below`` rows,
    IVF (nprobe 16) above. Returns (similarity, index), each ``n x k``.
    """
    import faiss

    faiss.omp_set_num_threads(threads)
    n, dim = vectors.shape
    index: Any
    if n <= exact_below:
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = int(4 * np.sqrt(n))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(0)
        index.train(vectors[rng.choice(n, size=min(n, 100_000), replace=False)])
        index.nprobe = 16
    index.add(vectors)
    sims, idx = index.search(vectors, k + 1)
    return _drop_self(sims, idx, k)


def _drop_self(sims: np.ndarray, idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove each row's own id (not always first when vectors tie) and keep ``k`` neighbours."""
    n = len(idx)
    is_self = idx == np.arange(n)[:, None]
    # rows where self was not retrieved (IVF miss) drop their last column instead
    no_self = ~is_self.any(axis=1)
    is_self[no_self, -1] = True
    keep = ~is_self
    return sims[keep].reshape(n, k), idx[keep].reshape(n, k)


def mutual_graph(
    sims: np.ndarray,
    idx: np.ndarray,
    *,
    min_sim: float = 0.0,
    adaptive_beta: float | None = None,
) -> sp.csr_matrix:
    """Symmetric weighted mutual-kNN graph, edge weight = cosine similarity.

    An edge i-j survives when each endpoint lists the other among its k neighbours. ``min_sim``
    is a global floor. ``adaptive_beta`` adds a per-node floor mean_i - beta * std_i of the node's
    neighbour similarities; an edge must clear the floors of both endpoints.
    """
    n, k = idx.shape
    w = sims.astype(np.float32).copy()
    w[w < min_sim] = 0.0
    if adaptive_beta is not None:
        floor = sims.mean(axis=1) - adaptive_beta * sims.std(axis=1)
        w[sims < floor[:, None]] = 0.0
    rows = np.repeat(np.arange(n), k)
    directed = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    directed.eliminate_zeros()
    mutual = directed.minimum(directed.T)  # zero unless both directions exist
    mutual.eliminate_zeros()
    return mutual.tocsr()


def upper_edges(graph: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = sp.triu(graph, k=1).tocoo()
    return coo.row, coo.col, coo.data


def _igraph(n: int, graph: sp.csr_matrix) -> tuple[Any, np.ndarray]:
    import igraph as ig

    r, c, w = upper_edges(graph)
    g = ig.Graph(n=n, edges=list(zip(r.tolist(), c.tolist(), strict=True)))
    return g, w.astype(float)


def leiden_cpm(
    graph: sp.csr_matrix,
    gamma: float,
    *,
    init: np.ndarray | None = None,
    seed: int = 0,
    n_iterations: int = 3,
    ig_graph: tuple[Any, np.ndarray] | None = None,
) -> np.ndarray:
    """Leiden with the Constant Potts Model. ``init`` (cluster code per node, -1 = own cluster)
    seeds the search; nodes may still move to any community."""
    import leidenalg as la

    g, w = ig_graph or _igraph(graph.shape[0], graph)
    start = None
    if init is not None:
        start = init.copy()
        free = start < 0
        start[free] = start.max() + 1 + np.arange(int(free.sum()))
        start = start.tolist()
    part = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights=w,
        resolution_parameter=gamma,
        initial_membership=start,
        n_iterations=n_iterations,
        seed=seed,
    )
    return np.asarray(part.membership, dtype=np.int64)


def leiden_modularity(graph: sp.csr_matrix, resolution: float, *, seed: int = 0) -> np.ndarray:
    import leidenalg as la

    g, w = _igraph(graph.shape[0], graph)
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=w,
        resolution_parameter=resolution,
        n_iterations=3,
        seed=seed,
    )
    return np.asarray(part.membership, dtype=np.int64)


def consensus_leiden(
    graph: sp.csr_matrix, gamma: float, *, runs: int = 5, tau: float = 0.5, seed: int = 0
) -> np.ndarray:
    """Consensus clustering (Lancichinetti and Fortunato): co-association of ``runs`` Leiden runs
    on the graph's own edges, edges below ``tau`` dropped, then one more Leiden on that graph."""
    pack = _igraph(graph.shape[0], graph)
    r, c, _ = upper_edges(graph)
    co = np.zeros(len(r), dtype=np.float32)
    for s in range(runs):
        m = leiden_cpm(graph, gamma, seed=seed + s, ig_graph=pack)
        co += m[r] == m[c]
    co /= runs
    keep = co >= tau
    n = graph.shape[0]
    cons = sp.csr_matrix((co[keep], (r[keep], c[keep])), shape=(n, n))
    cons = cons + cons.T
    return leiden_cpm(cons, gamma, seed=seed + runs)


def codes(*labels: np.ndarray) -> np.ndarray:
    """Dense code of the tuple of label arrays (the intersection of partitions)."""
    key = np.zeros(len(labels[0]), dtype=np.int64)
    for lab in labels:
        key = key * (int(lab.max()) + 2) + (lab.astype(np.int64) + 1)
    return np.unique(key, return_inverse=True)[1].astype(np.int64)


def chameleon_merge(
    graph: sp.csr_matrix, atoms: np.ndarray, *, ri_min: float, rc_min: float
) -> np.ndarray:
    """CHAMELEON-style merge of ``atoms`` (a fine partition) by relative interconnectivity and
    relative closeness, best pair first.

    RI = (cross weight per node of the smaller side) / (mean internal weighted degree of the two
    sides); RC = mean cross edge weight / size-weighted mean internal edge weight. Internal
    statistics are totals, not min-cut bisections, so this is the usual cheap approximation.
    Two atoms merge only when both ratios clear their floors; stats are recomputed after each
    merge. Atoms that never validate stay apart, which is how an ambiguous move is refused.
    """
    r, c, w = upper_edges(graph)
    a, b = atoms[r], atoms[c]
    m = int(atoms.max()) + 1
    size = np.bincount(atoms, minlength=m).astype(np.float64)
    inside = a == b
    e_in = np.bincount(a[inside], weights=w[inside], minlength=m)
    c_in = np.bincount(a[inside], minlength=m).astype(np.float64)
    global_mean = float(w.mean()) if len(w) else 1.0
    global_deg = 2 * float(w.sum()) / max(1, len(atoms))
    cross: dict[int, dict[int, list[float]]] = {}
    for x, y, wt in zip(a[~inside].tolist(), b[~inside].tolist(), w[~inside].tolist(), strict=True):
        for p, q in ((x, y), (y, x)):
            cell = cross.setdefault(p, {}).setdefault(q, [0.0, 0.0])
            cell[0] += wt
            cell[1] += 1.0

    def score(x: int, y: int) -> float:
        sw, sc = cross[x][y]
        ref = [2 * e_in[t] / size[t] for t in (x, y) if size[t] > 1 and c_in[t] > 0]
        s_ref = float(np.mean(ref)) if ref else global_deg
        ri = (sw / min(size[x], size[y])) / max(s_ref, 1e-9)
        means = [(size[t], e_in[t] / c_in[t] if c_in[t] else global_mean) for t in (x, y)]
        m_ref = sum(sz * mu for sz, mu in means) / (size[x] + size[y])
        rc = (sw / sc) / max(m_ref, 1e-9)
        return ri * rc if ri >= ri_min and rc >= rc_min else 0.0

    version = np.zeros(m, dtype=np.int64)
    alive = np.ones(m, dtype=bool)
    parent = np.arange(m)
    heap: list[tuple[float, int, int, int, int]] = []
    heap.extend(
        (-s, x, y, 0, 0) for x, nb in cross.items() for y in nb if x < y and (s := score(x, y)) > 0
    )
    heapq.heapify(heap)
    while heap:
        _, x, y, vx, vy = heapq.heappop(heap)
        if not (alive[x] and alive[y]) or version[x] != vx or version[y] != vy:
            continue
        sw, sc = cross[x].pop(y)
        cross[y].pop(x)
        e_in[x] += e_in[y] + sw
        c_in[x] += c_in[y] + sc
        size[x] += size[y]
        alive[y] = False
        parent[y] = x
        for z, (zw, zc) in cross.pop(y).items():
            cross[z].pop(y)
            cell = cross[x].setdefault(z, [0.0, 0.0])
            cell[0] += zw
            cell[1] += zc
            cross[z][x] = cell
        version[x] += 1
        for z in cross[x]:
            if (s := score(x, z)) > 0:
                lo, hi = (x, z) if x < z else (z, x)
                heapq.heappush(heap, (-s, lo, hi, int(version[lo]), int(version[hi])))
    root = parent.copy()
    while True:  # path compression to the surviving atom
        nxt = root[root]
        if (nxt == root).all():
            break
        root = nxt
    return np.unique(root[atoms], return_inverse=True)[1].astype(np.int64)


def apply_min_size(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Clusters smaller than ``min_size`` become unassigned (-1); the rest are renumbered."""
    labels = labels.copy()
    sizes = np.bincount(labels[labels >= 0])
    labels[(labels >= 0) & (sizes[np.maximum(labels, 0)] < min_size)] = UNASSIGNED
    ok = labels >= 0
    labels[ok] = np.unique(labels[ok], return_inverse=True)[1]
    return labels
