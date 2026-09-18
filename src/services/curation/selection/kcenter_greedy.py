"""k-center-greedy core-set selection (curation-strategy plan §2.6/§3.4).

Sener & Savarese, "Active Learning for Convolutional Neural Networks: A
Core-Set Approach" (ICLR 2018) — the standard greedy k-center algorithm.
This is the real technique behind "diversity sampling" / "subpart
diversity" in tools like LightlyStudio (plan §2.6). Produces a selection
order, **never** a ``cluster_id`` — it's an overlay, not an assignment
(plan §0/§8 non-goal #3). This module is pure math: no OpenSearch, no
faiss, no I/O of any kind, so it's directly unit-testable.

**Why plain numpy, not faiss (house convention check, plan §3.4):**
:mod:`crop_scores.uniqueness` uses ``faiss.IndexIVFFlat`` because its
workload is an *approximate nearest-neighbour search* (k=16 neighbours
out of ~125k candidates per query, repeated for every point) — exactly
the access pattern FAISS's IVF index is built to accelerate. k-center-
greedy's access pattern is different: each of the ``k`` iterations needs
an **exact** running-min-distance argmax over *every remaining point*
(not a top-k neighbour search), which is a single dense ``(n, d) @ (d,)``
matrix-vector product per iteration — already BLAS-backed and optimal
via plain ``numpy``. There is no approximate-search shortcut for "find
the point that is farthest from the selected set so far" the way there
is for "find this point's 16 nearest neighbours", so introducing faiss
here would add complexity without a speed win.

**Compute shape:** O(n·k·d) — n full-pool distance updates, k times.
When ``k == n`` (a full-pool ranking, as ``legacy_select.py`` needs for
``GET /curation/crops?order=diverse`` pagination) this degrades to O(n²·d),
which is why that caller uses a much smaller inline-pool cap than the
POST endpoint's k-bounded selection — see ``legacy_select.py`` module
docstring for the exact budget derivation.
"""

from __future__ import annotations

import numpy as np


def k_center_greedy(x: np.ndarray, k: int, *, seed_idx: int | None = None) -> np.ndarray:
    """Return ``k`` row indices of ``x``, greedily maximizing min-distance coverage.

    ``x`` MUST be L2-normalized (cosine similarity == dot product for unit
    vectors), so cosine distance is computed as ``1 - x @ x_i`` without a
    second norm pass per iteration.

    Deterministic given the same ``x`` and ``seed_idx`` — there is no
    randomness anywhere in this function; every tie in the running argmax
    is broken by ``np.argmax``'s own "first occurrence wins" convention.

    Args:
        x: ``(n, d)`` array, L2-normalized rows (float32 recommended;
            cast internally regardless).
        k: number of points to select. Clamped to ``min(k, n)``.
        seed_idx: row index of the first point to select. If ``None``,
            the first point is chosen as the point **farthest from the
            batch centroid** (lowest cosine similarity to the mean
            direction), not a fixed index like ``0``. Rationale: a real
            embedding pool is dominated by dense, redundant regions (the
            common vehicle types); starting the greedy walk at a fixed
            index risks starting *inside* that dense mass (whichever
            direction happens to sort first), wasting the first pick on
            a point that free-rides on later coverage anyway. Starting
            at the point most unlike the "average" crop guarantees the
            very first selection is informative, and it's still 100%
            deterministic for a given ``x`` (no RNG involved) — this
            mirrors the throwaway validation implementation that produced
            the 1.50x pre-screen result in
            ``docs/design/curation_scores.md`` §6.

    Returns:
        ``(min(k, n),)`` int64 array of selected row indices, in selection
        order (index 0 is the seed). Empty array if ``x`` has 0 rows or
        ``k <= 0``.
    """
    n = x.shape[0]
    if n == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64)
    k = min(k, n)

    xf = np.ascontiguousarray(x, dtype=np.float32)

    if seed_idx is None:
        centroid = xf.mean(axis=0)
        cnorm = float(np.linalg.norm(centroid))
        if cnorm > 0:
            centroid = centroid / cnorm
        sims_to_centroid = xf @ centroid
        first = int(np.argmin(sims_to_centroid))
    else:
        if not (0 <= seed_idx < n):
            raise ValueError(f'seed_idx {seed_idx} out of range for n={n}')
        first = int(seed_idx)

    selected = np.empty(k, dtype=np.int64)
    selected[0] = first

    # min_dist[i] = cosine distance from row i to the nearest already-
    # selected row. -inf sentinel on already-selected rows keeps them out
    # of every future argmax permanently (np.minimum never raises a value
    # back up once it's been pushed to -inf).
    min_dist = 1.0 - (xf @ xf[first])
    min_dist[first] = -np.inf

    for step in range(1, k):
        nxt = int(np.argmax(min_dist))
        selected[step] = nxt
        d = 1.0 - (xf @ xf[nxt])
        np.minimum(min_dist, d, out=min_dist)
        min_dist[nxt] = -np.inf

    return selected


__all__ = ['k_center_greedy']
