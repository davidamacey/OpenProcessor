"""Whole-frame near-duplicate grouping over whole-frame embeddings.

Pure-numpy grouping kept only for the 0.6.0 near-duplicate scorer under
``item_scores/``; the export reads the stored groups of the current run instead.

Method: the whole-frame embeddings are unit-norm, so a dot product is
cosine similarity. We form an undirected graph linking any pair with cosine
``>= threshold`` and take connected components (union-find) as near-dup groups.
The cosine pass is exact (full ``X·Xᵀ`` in row blocks) — deterministic, which
matters for a reproducible frozen dataset.

Threshold convention: **0.98** is the training cut. At/above 0.98 frames
are true duplicates (same subject, same angle, same moment) and collapse to
one representative; below that (0.96-0.97 = same subject, different angle)
they are KEPT as useful viewpoint variation, not redundancy.
"""

from __future__ import annotations

import numpy as np


DEFAULT_FRAME_DEDUP_THRESHOLD = 0.98


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self._p
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[ra] = rb


def near_dup_groups(
    mat: np.ndarray, threshold: float = DEFAULT_FRAME_DEDUP_THRESHOLD, block: int = 2048
) -> list[list[int]]:
    """Return connected-component groups (size >= 2) of near-duplicate rows.

    Args:
        mat: ``(n, d)`` float32 array of unit-norm embeddings (dot == cosine).
        threshold: cosine cutoff; pairs ``>= threshold`` are linked.
        block: row-block size for the chunked ``X·Xᵀ`` pass (memory knob).

    Returns:
        Groups of row indices that are mutually near-duplicate (transitively),
        each of length >= 2. Singletons are omitted.
    """
    n = mat.shape[0]
    if n < 2:
        return []
    uf = _UnionFind(n)
    mat_t = mat.T
    for r0 in range(0, n, block):
        r1 = min(r0 + block, n)
        sims = mat[r0:r1] @ mat_t  # (b, n) cosine
        rows, cols = np.where(sims >= threshold)
        gi = rows + r0
        keep = cols > gi  # upper triangle only; drops self-pairs + double counts
        for a, b in zip(gi[keep], cols[keep], strict=True):
            uf.union(int(a), int(b))
    roots: dict[int, list[int]] = {}
    for i in range(n):
        roots.setdefault(uf.find(i), []).append(i)
    return [g for g in roots.values() if len(g) > 1]


def most_central(mat: np.ndarray, group: list[int]) -> int:
    """Index (into ``group``) of the most central member (max summed cosine).

    The representative kept when a near-dup group is collapsed — the frame most
    typical of the burst, rather than an arbitrary one.
    """
    sub = mat[group]
    centrality = (sub @ sub.T).sum(axis=1)
    return int(np.argmax(centrality))
