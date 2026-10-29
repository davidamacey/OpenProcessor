"""Grouping methods inside the car key: partitions of a similarity matrix, one code per row.

Average linkage, k-reciprocal re-ranking and DBA come from ``scripts/research/cluster_diag``;
this module adds the ECG-style ensemble and the thresholded cuts used to trace an operating curve.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster

from scripts.research.cluster_diag.diagnostics import (
    dba_similarity,
    kreciprocal_similarity,
    leiden_cpm_partition,
    linkage_of,
    mutual_knn_graph,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]

__all__ = [
    'ahc_cuts',
    'cos_sim',
    'dba_similarity',
    'ecg_edge_weights',
    'ecg_partition',
    'kreciprocal_similarity',
    'leiden_cpm_partition',
]

ECG_ENSEMBLE = 32
ECG_MIN_WEIGHT = 0.05


def cos_sim(vectors: Floats) -> Floats:
    v = np.asarray(vectors, dtype=np.float32)
    return v @ v.T


def ahc_cuts(
    sim: Floats, heights: Sequence[float], method: str = 'average'
) -> tuple[Floats, list[Ints]]:
    """The dendrogram heights and the partitions cut at each of ``heights`` (distance ``max - sim``)."""
    z = linkage_of(sim, method)
    return z[:, 2], [fcluster(z, h, 'distance') - 1 for h in heights]


def ecg_edge_weights(
    n_edges_in_runs: Sequence[Sequence[bool]], min_weight: float = ECG_MIN_WEIGHT
) -> list[float]:
    """ECG edge weights: ``min_weight + (1 - min_weight) * share of ensemble runs with the edge inside a community``.

    ``n_edges_in_runs[r][e]`` is True when run ``r`` puts both ends of edge ``e`` in one community.
    """
    runs = np.asarray(n_edges_in_runs, dtype=np.float64)
    return [float(x) for x in min_weight + (1 - min_weight) * runs.mean(axis=0)]


def ecg_partition(
    sim: Floats, k: int = 30, seed: int = 0, ensemble: int = ECG_ENSEMBLE, min_size: int = 3
) -> Ints:
    """ECG (Poulin and Theberge 2019) on the mutual k-NN graph: first-level Louvain ensemble, then Louvain.

    Each ensemble run shuffles the vertex order with its own seed; clusters under ``min_size`` are
    left UNASSIGNED. igraph's python binding has no ECG, so this follows the paper's definition.
    """
    import igraph as ig

    from scripts.bench.cluster_eval_metrics import UNASSIGNED

    edges, raw = mutual_knn_graph(sim, k)
    n = len(sim)
    w0 = [x - float(sim.min()) for x in raw]
    graph = ig.Graph(n=n, edges=edges)
    graph.es['weight'] = w0
    in_runs = []
    for r in range(ensemble):
        rng = random.Random(f'ecg:{seed}:{r}')
        perm = list(range(n))
        rng.shuffle(perm)
        ig.set_random_number_generator(random.Random(f'ecg-run:{seed}:{r}'))
        shuffled = ig.Graph(n=n, edges=[(perm[a], perm[b]) for a, b in edges])
        shuffled.es['weight'] = w0
        first = shuffled.community_multilevel(weights='weight', return_levels=True)[0].membership
        in_runs.append([first[perm[a]] == first[perm[b]] for a, b in edges])
    graph.es['weight'] = [a * b for a, b in zip(w0, ecg_edge_weights(in_runs), strict=True)]
    ig.set_random_number_generator(random.Random(f'ecg-final:{seed}'))
    part = np.array(graph.community_multilevel(weights='weight').membership, dtype=np.int64)
    part[np.bincount(part)[part] < min_size] = UNASSIGNED
    return part
