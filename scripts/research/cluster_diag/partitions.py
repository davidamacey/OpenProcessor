"""Partition producers for the configuration baseline (E1) and the class-prior test (E5).

Every function maps unit vectors (or a similarity matrix) to one integer cluster code per row,
``UNASSIGNED`` (-1) for a row no cluster claimed, so all arms go through the same scorer.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
from scipy.cluster.hierarchy import fcluster

from scripts.bench.cluster_eval_metrics import UNASSIGNED
from scripts.research.cluster_diag.diagnostics import (
    linkage_from_distance,
    linkage_of,
    mutual_knn_graph,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]
    Labels = Sequence[Any] | NDArray[Any]

MIN_CLUSTER = 3
REGION_BUCKET_SIZE = 800  # region_box_clustering.REGION_TARGET_BUCKET_SIZE


def ivf_current(vectors: Floats) -> Ints:
    """The production residual-pool clusterer (``IVFMethod``, K = min(512, n // 6)) on the CPU.

    Centroids go to a temporary state directory so no project directory is touched.
    """
    from src.services.curation.clustering.methods import ivf_store
    from src.services.curation.clustering.methods.ivf import IVFMethod

    with tempfile.TemporaryDirectory() as state:
        cfg = SimpleNamespace(project_state_dir=Path(state))
        with mock.patch.object(ivf_store, 'get_curation_config', return_value=cfg):
            result = asyncio.run(
                IVFMethod().fit_predict(
                    np.ascontiguousarray(vectors, dtype=np.float32),
                    backend_info=SimpleNamespace(name='cpu'),  # type: ignore[arg-type]
                )
            )
    return np.asarray(result.labels, dtype=np.int64)


def region_kmeans(vectors: Floats) -> Ints:
    """The 2b region-box clusterer: MiniBatchKMeans, K = min(max(8, round(n / 800)), n)."""
    from sklearn.cluster import MiniBatchKMeans

    n = len(vectors)
    k = min(max(8, round(n / REGION_BUCKET_SIZE)), n)
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096)
    return np.asarray(km.fit_predict(vectors), dtype=np.int64)


def ahc_cuts(
    sim: Floats, method: str, n_cuts: int = 60, lo: float = 0.5, hi: float = 0.995
) -> list[tuple[float, Ints]]:
    """``(cut height, partition)`` pairs along one dendrogram, at quantiles of its heights."""
    z = linkage_of(sim, method)
    heights = np.quantile(z[:, 2], np.linspace(lo, hi, n_cuts))
    return [(float(t), fcluster(z, t, 'distance').astype(np.int64) - 1) for t in heights]


def drop_small(part: Ints, min_size: int = MIN_CLUSTER) -> Ints:
    """Copy of ``part`` with clusters under ``min_size`` set to ``UNASSIGNED``."""
    out = np.array(part, dtype=np.int64)
    assigned = out >= 0
    sizes = np.bincount(out[assigned], minlength=int(out.max()) + 1 if assigned.any() else 0)
    small = np.zeros(len(out), dtype=bool)
    small[assigned] = sizes[out[assigned]] < min_size
    out[small] = UNASSIGNED
    return out


def ecg_partition(
    sim: Floats,
    k: int = 30,
    n_runs: int = 16,
    min_weight: float = 0.05,
    min_size: int = MIN_CLUSTER,
    seed: int = 0,
) -> Ints:
    """Ensemble clustering for graphs (Poulin and Theberge 2019) on the mutual k-NN graph.

    ``n_runs`` Louvain first-level partitions of randomly relabelled copies of the graph give
    each edge the share of runs that put its ends together; the final Louvain pass uses
    ``min_weight + (1 - min_weight) * share`` as the edge weight. The python-igraph in the
    research environment has no ``community_ecg``, hence this implementation.
    """
    import random

    import igraph as ig

    ig.set_random_number_generator(random.Random(seed))  # igraph draws from this, not numpy
    edges, weights = mutual_knn_graph(sim, k)
    n = len(sim)
    graph = ig.Graph(n=n, edges=edges)
    rng = np.random.default_rng(seed)
    together = np.zeros(len(edges))
    ends = np.array(edges, dtype=np.int64).reshape(-1, 2)
    for _ in range(n_runs):
        perm = rng.permutation(n)  # vertex i becomes perm[i]
        base = graph.permute_vertices(perm.tolist())
        level = base.community_multilevel(weights=weights, return_levels=True)[0].membership
        member = np.asarray(level)[perm]  # membership back in original vertex order
        together += member[ends[:, 0]] == member[ends[:, 1]]
    ecg_weights = min_weight + (1 - min_weight) * together / n_runs
    final = graph.community_multilevel(weights=ecg_weights.tolist())
    return drop_small(np.asarray(final.membership, dtype=np.int64), min_size)


def group_linkages(
    groups: Labels, sim: Floats, method: str = 'average'
) -> list[tuple[Ints, Floats | None]]:
    """One dendrogram per group (``None`` for a single row), on the global ``sim.max() - sim``."""
    g = np.asarray(groups).astype(str)
    out: list[tuple[Ints, Floats | None]] = []
    for name in sorted(set(g.tolist())):
        idx = np.nonzero(g == name)[0]
        z = (
            None
            if len(idx) == 1
            else linkage_from_distance(sim.max() - sim[np.ix_(idx, idx)], method)
        )
        out.append((idx, z))
    return out


def cut_groups(n: int, linkages: Sequence[tuple[Ints, Floats | None]], cut: float) -> Ints:
    """Cut every group's dendrogram at one height; cluster codes are unique across groups."""
    labels = np.full(n, UNASSIGNED, dtype=np.int64)
    offset = 0
    for idx, z in linkages:
        part = np.zeros(len(idx), dtype=np.int64) if z is None else fcluster(z, cut, 'distance') - 1
        labels[idx] = part + offset
        offset += int(part.max()) + 1
    return labels
