"""Face grouping: the one place that turns vectors into connected components."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np


def similarity_components(
    embeddings: np.ndarray, threshold: float, *, batch: int = 4096
) -> list[list[int]]:
    """Connected components of the graph with an edge wherever cosine similarity >= threshold.

    This is the grouping interface of person clustering: normalised-by-this-function
    vectors and a threshold in, components out (every row appears in exactly one
    group, singletons included; members ascending, groups ordered by first member).
    The stable-id, lock and write logic only consume its result, so a faster
    approximate or GPU method (face-grouping research, issue #346) can replace the
    body without touching them.

    The implementation is exact: a flat inner-product range search, so the partition
    equals the one a dense ``x @ x.T >= threshold`` produces. Memory is the matrix
    held twice (array plus index), O(n) component state and one batch of matches
    (12 bytes each; a batch where every face is alike is ``batch * n`` matches).
    Compute is O(n^2 d): about 4 minutes at 200k x 512 on 8 threads.
    """
    import faiss
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(embeddings)
    if n == 0:
        return []
    x = np.array(embeddings, dtype=np.float32, order='C', copy=True)
    faiss.normalize_L2(x)
    faiss.omp_set_num_threads(min(8, os.cpu_count() or 1))
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    # range_search keeps scores strictly above the radius; step down one float32 ulp so
    # a pair exactly at the threshold is an edge, as in `>=`.
    radius = float(np.nextafter(np.float32(threshold), np.float32(-np.inf)))

    rows_kept = np.empty(0, dtype=np.int64)
    cols_kept = np.empty(0, dtype=np.int64)
    labels = np.arange(n)
    for start in range(0, n, batch):
        lims, _, nbr = index.range_search(x[start : start + batch], radius)
        bounds = lims.astype(np.int64)
        rows = np.repeat(
            np.arange(start, start + len(lims) - 1, dtype=np.int64), bounds[1:] - bounds[:-1]
        )
        keep = nbr.astype(np.int64) > rows
        r = np.concatenate([rows_kept, rows[keep]])
        c = np.concatenate([cols_kept, nbr[keep].astype(np.int64)])
        graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
        # Replace the edges by a star per component (<= n edges): same reachability.
        rep = np.full(labels.max() + 1, n, dtype=np.int64)
        np.minimum.at(rep, labels, np.arange(n))
        nodes = np.flatnonzero(rep[labels] != np.arange(n))
        rows_kept, cols_kept = nodes, rep[labels[nodes]]

    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels.tolist()):
        groups.setdefault(label, []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])
