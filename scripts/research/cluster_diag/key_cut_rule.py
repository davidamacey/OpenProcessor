"""A label-free cut per detector key (Y2): alpha times the key's median 10th-neighbour distance.

The scale of cosine distance differs between projects and between keys (set P's vectors are
tighter than the public set's). The rule reads that scale from the key's own crops, so one factor
``alpha`` fitted on a labelled calibration half can set a cut on a key that has no labels. A key
with too few crops for a 10th neighbour has no rule cut: it is refused, never given another key's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench.cluster_eval_metrics import UNASSIGNED


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]

NEIGHBOUR = 10
CHUNK = 2048


def nth_neighbour_distance(vectors: Floats, k: int = NEIGHBOUR, chunk: int = CHUNK) -> Floats:
    """Cosine distance (1 - dot, floored at 0) from each row to its ``k``-th nearest other row."""
    n = len(vectors)
    if n <= k:
        raise ValueError(f'{n} rows have no {k}-th neighbour')
    out = np.empty(n, dtype=np.float64)
    for i0 in range(0, n, chunk):
        sims = vectors[i0 : i0 + chunk] @ vectors.T
        sims[np.arange(sims.shape[0]), np.arange(i0, i0 + sims.shape[0])] = -np.inf
        kth = np.partition(-sims, k - 1, axis=1)[:, k - 1]  # k-th largest similarity
        out[i0 : i0 + chunk] = np.clip(1.0 + kth, 0.0, None)
    return out


def key_scale(vectors: Floats, k: int = NEIGHBOUR) -> float | None:
    """Median k-th neighbour distance of one key, or ``None`` when it has no k-th neighbour."""
    if len(vectors) <= k:
        return None
    return float(np.median(nth_neighbour_distance(vectors, k)))


def key_scales(vectors: Floats, keys: Any, k: int = NEIGHBOUR) -> dict[str, float | None]:
    """``key_scale`` of every key."""
    kk = np.asarray(keys).astype(str)
    return {name: key_scale(vectors[kk == name], k) for name in sorted(set(kk.tolist()))}


def rule_cuts(scales: dict[str, float | None], alpha: float) -> dict[str, float | None]:
    """The cut each key gets at factor ``alpha`` (``None`` for a refused key)."""
    return {k: None if s is None else alpha * s for k, s in scales.items()}


def partition_with_cuts(
    n: int,
    keys: Any,
    cuts: dict[str, float | None],
    cut_key: Callable[[str, float], Ints],
) -> Ints:
    """Cluster codes with a per-key cut; a refused key's crops are unassigned.

    ``cut_key(name, height)`` returns the cluster codes (0-based) of the rows of key ``name`` in
    row order. Codes are unique across keys.
    """
    kk = np.asarray(keys).astype(str)
    labels = np.full(n, UNASSIGNED, dtype=np.int64)
    offset = 0
    for name in sorted(set(kk.tolist())):
        cut = cuts.get(name)
        if cut is None:
            continue
        idx = np.nonzero(kk == name)[0]
        part = cut_key(name, cut)
        labels[idx] = part + offset
        offset += int(part.max()) + 1
    return labels
