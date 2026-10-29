"""Embedding-geometry and neighbour-purity diagnostics for crop vectors (research, CPU only).

Pure numpy and scipy over item-aligned arrays; nothing here touches OpenSearch. The functions
are the ones the consolidated clustering note (section 4) was computed with:

* geometry of the space: random-pair cosine, PCA spectrum (effective rank), k-occurrence
  hubness;
* neighbour purity: P@k and the k-NN vote under a similarity matrix, split by stratum;
* similarity transforms: PCA whitening fitted on an independent pool, CSLS local scaling,
  k-reciprocal re-ranking (Zhong et al. 2017) and database-side augmentation;
* clustering curves: average- and complete-linkage AHC cuts and Leiden CPM sweeps, scored with
  the repo's pair metrics so a figure here means what it means in the evaluation harness.

A similarity matrix ``S`` is ``(n, n)`` with larger meaning closer; a distance is
``S.max() - S``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import skew

from scripts.bench.cluster_eval_metrics import UNASSIGNED, pairwise_scores


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]
    Labels = Sequence[Any] | NDArray[Any]

RECALL_FLOORS = (0.1, 0.2, 0.3, 0.5)


def l2_normalize(x: Floats) -> Floats:
    """Unit-norm rows (a zero row stays zero)."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.where(norm == 0, 1.0, norm)).astype(np.float32)


# ---------------------------------------------------------------- geometry of the space


def pca_spectrum(vectors: Floats) -> Floats:
    """Eigenvalues of the covariance matrix, largest first, negatives from rounding clipped."""
    centred = vectors - vectors.mean(0)
    ev = np.linalg.eigvalsh(centred.T @ centred / len(vectors))
    return np.clip(np.sort(ev)[::-1], 0.0, None)


def effective_rank(vectors: Floats) -> float:
    """exp(entropy of the normalised PCA spectrum): the number of directions that carry variance."""
    ev = pca_spectrum(vectors)
    p = ev[ev > 0] / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def top_k_neighbours(sim: Floats, k: int) -> Ints:
    """Indices of the ``k`` most similar other rows per row, best first (self excluded)."""
    masked = sim.astype(np.float64, copy=True)
    np.fill_diagonal(masked, -np.inf)
    part = np.argpartition(-masked, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(masked, part, 1), axis=1, kind='stable')
    return np.take_along_axis(part, order, 1)


def k_occurrence(neigh: Ints, n: int) -> Ints:
    """How many rows list each row among their k nearest neighbours (N_k)."""
    return np.bincount(neigh.ravel(), minlength=n)


def hubness(sim: Floats, k: int = 10) -> dict[str, float]:
    """Skewness of N_k, share of anti-hubs (never anyone's neighbour) and the largest N_k."""
    nk = k_occurrence(top_k_neighbours(sim, k), len(sim))
    return {
        f'hub_skew_N{k}': float(skew(nk)),
        f'antihub_frac_N{k}': float((nk == 0).mean()),
        f'max_N{k}': int(nk.max()),
        'top1pct_share_of_occurrences': float(
            np.sort(nk)[::-1][: max(1, len(nk) // 100)].sum() / nk.sum()
        ),
    }


def geometry(vectors: Floats, k: int = 10, n_pairs: int = 200_000, seed: int = 0) -> dict[str, Any]:
    """Anisotropy (random-pair cosine, norm of the mean), spectrum and hubness of unit rows."""
    n = len(vectors)
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = a != b
    cos = np.einsum('ij,ij->i', vectors[a[keep]], vectors[b[keep]])
    ev = pca_spectrum(vectors)
    share = ev / ev.sum()
    sim = vectors @ vectors.T
    neigh = top_k_neighbours(sim, k)
    nn1 = np.take_along_axis(sim, neigh[:, :1], 1)[:, 0]
    nnk = np.take_along_axis(sim, neigh[:, -1:], 1)[:, 0]
    return {
        'n': n,
        'mean_random_cos': float(cos.mean()),
        'std_random_cos': float(cos.std()),
        'p05_p95_random_cos': [float(np.percentile(cos, 5)), float(np.percentile(cos, 95))],
        'norm_of_mean': float(np.linalg.norm(vectors.mean(0))),
        'pc1_share': float(share[0]),
        'top10_share': float(share[:10].sum()),
        'effective_rank': effective_rank(vectors),
        'nn1_cos_median': float(np.median(nn1)),
        f'nn{k}_cos_median': float(np.median(nnk)),
        **hubness(sim, k),
    }


# ---------------------------------------------------------------- neighbour purity


def neighbour_hits(sim: Floats, truth: Labels, k: int = 10) -> NDArray[np.bool_]:
    """``hit[i, j]`` is True when the j-th nearest other row of ``i`` shares ``i``'s class."""
    y = np.asarray(truth)
    return np.asarray(y[top_k_neighbours(sim, k)] == y[:, None])


def knn_vote_accuracy(sim: Floats, truth: Labels, k: int = 10) -> float:
    """Share of rows whose class equals the most common class among their k neighbours.

    A tie goes to the class seen first among the neighbours, nearest first.
    """
    y = np.asarray(truth)
    neigh = top_k_neighbours(sim, k)
    correct = 0
    for i, row in enumerate(neigh):
        votes = Counter(y[row].tolist())
        top = max(votes.values())
        winner = next(c for c in y[row].tolist() if votes[c] == top)
        correct += winner == y[i]
    return correct / len(y)


def knn_report(
    sim: Floats, truth: Labels, strata: Mapping[str, NDArray[np.bool_]], k: int = 10
) -> dict[str, float]:
    """P@1, P@5, P@k, k-NN vote accuracy and P@k inside each boolean stratum mask."""
    hit = neighbour_hits(sim, truth, k)
    out = {
        'P@1': float(hit[:, 0].mean()),
        'P@5': float(hit[:, :5].mean()),
        f'P@{k}': float(hit.mean()),
        f'{k}NN_vote_acc': knn_vote_accuracy(sim, truth, k),
    }
    for name, mask in strata.items():
        if mask.any():
            out[f'P@{k}[{name}]'] = float(hit[mask].mean())
    return out


# ---------------------------------------------------------------- similarity transforms


def pca_whiten_fit(pool: Floats, dim: int, power: float = 0.5) -> tuple[Floats, Floats]:
    """Mean and projection of a PCA fitted on ``pool``: top ``dim`` axes scaled by eigval**-power.

    ``power`` 0.5 is full whitening (unit variance per axis), 0 is plain PCA.
    """
    mu = pool.mean(0)
    centred = pool - mu
    ev, vec = np.linalg.eigh(centred.T @ centred / len(pool))
    order = np.argsort(ev)[::-1][:dim]
    return mu, vec[:, order] / (np.clip(ev[order], 0, None) ** power + 1e-6)


def pca_whiten_apply(vectors: Floats, mu: Floats, proj: Floats) -> Floats:
    """Project with a fitted whitening and re-normalise to unit rows."""
    return l2_normalize((vectors - mu) @ proj)


def csls_similarity(vectors: Floats, reference: Floats, k: int = 10) -> Floats:
    """CSLS: 2 cos(x, y) - r(x) - r(y), r = mean cosine to the k nearest rows of ``reference``."""
    r = np.sort(vectors @ reference.T, axis=1)[:, -k:].mean(1)
    return 2 * (vectors @ vectors.T) - r[:, None] - r[None, :]


def _k_reciprocal(order: Ints, i: int, k: int) -> Ints:
    forward = order[i, : k + 1]
    back = order[forward, : k + 1]
    return forward[np.any(back == i, axis=1)]


def kreciprocal_similarity(vectors: Floats, k1: int = 20, k2: int = 6, lam: float = 0.3) -> Floats:
    """Negated k-reciprocal re-ranked distance (Zhong et al. 2017) over the set itself."""
    n = len(vectors)
    dist = 2 - 2 * (vectors @ vectors.T)
    order = np.argsort(dist, axis=1, kind='stable')
    weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        rec = _k_reciprocal(order, i, k1)
        expanded = rec
        for j in rec:
            rec_j = _k_reciprocal(order, int(j), round(k1 / 2))
            if len(np.intersect1d(rec_j, rec)) > 2 / 3 * len(rec_j):
                expanded = np.union1d(expanded, rec_j)
        w = np.exp(-dist[i, expanded])
        weights[i, expanded] = w / w.sum()
    if k2 > 1:
        weights = np.stack([weights[order[i, :k2]].mean(0) for i in range(n)])
    holders = [np.nonzero(weights[:, j])[0] for j in range(n)]
    jaccard = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        overlap = np.zeros(n, dtype=np.float32)
        for j in np.nonzero(weights[i])[0]:
            overlap[holders[j]] += np.minimum(weights[i, j], weights[holders[j], j])
        jaccard[i] = 1 - overlap / (2 - overlap)
    return -(jaccard * (1 - lam) + (dist / 4) * lam)


def dba_similarity(vectors: Floats, k: int = 5, alpha: float = 3.0) -> Floats:
    """Cosine after replacing each row by the cos**alpha weighted mean of its k nearest rows.

    The row itself is one of its own neighbours (cosine 1), as in alpha-query-expansion.
    """
    sim = vectors @ vectors.T
    idx = np.argsort(-sim, axis=1, kind='stable')[:, :k]
    w = np.take_along_axis(sim, idx, 1).clip(0) ** alpha
    z = l2_normalize(np.einsum('nk,nkd->nd', w, vectors[idx]))
    return z @ z.T


# ---------------------------------------------------------------- clustering curves

CurvePoint = tuple[
    float, float | None, float | None, float
]  # threshold, precision, recall, unassigned


def score_partition(partition: Ints, truth: Labels, labelled: Ints) -> CurvePoint:
    """Pair precision, recall and unassigned share of ``partition`` over the labelled rows only."""
    part = np.asarray(partition)[labelled]
    s = pairwise_scores(np.asarray(truth), part)
    return (math.nan, s['pair_precision'], s['pair_recall'], float(s['unassigned_fraction'] or 0.0))


def _threshold_curve(
    dendrogram: Floats, truth: Labels, labelled: Ints, n_points: int
) -> list[CurvePoint]:
    cuts = np.quantile(dendrogram[:, 2], np.linspace(0.5, 0.995, n_points))
    curve = []
    for t in cuts:
        part = fcluster(dendrogram, t, 'distance') - 1
        curve.append((float(t), *score_partition(part, truth, labelled)[1:]))
    return curve


def linkage_from_distance(dist: Floats, method: str) -> Floats:
    """Scipy linkage of a square distance matrix (symmetrised, zero diagonal)."""
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    return np.asarray(linkage(squareform(dist, checks=False), method=method))


def linkage_of(sim: Floats, method: str) -> Floats:
    """Scipy linkage over the distance ``sim.max() - sim``."""
    return linkage_from_distance(sim.max() - sim, method)


def ahc_curve(
    sim: Floats,
    truth: Labels,
    labelled: Ints,
    method: str = 'average',
    n_points: int = 40,
) -> list[CurvePoint]:
    """Pair precision/recall along dendrogram cuts at the 50th to 99.5th percentile of heights."""
    return _threshold_curve(linkage_of(sim, method), truth, labelled, n_points)


def mutual_knn_graph(sim: Floats, k: int) -> tuple[list[tuple[int, int]], list[float]]:
    """Edges (i < j) where each end lists the other among its k nearest, weighted by ``sim``."""
    neigh = top_k_neighbours(sim, k)
    sets = [set(row.tolist()) for row in neigh]
    edges, weights = [], []
    for i, row in enumerate(neigh):
        for j in sorted(row.tolist()):
            if i < j and i in sets[j]:
                edges.append((i, j))
                weights.append(float(sim[i, j]))
    return edges, weights


def leiden_cpm_partition(
    sim: Floats, k: int, fraction: float, min_size: int = 3, seed: int = 0
) -> Ints:
    """Leiden with the CPM quality on the mutual k-NN graph; gamma = ``fraction`` * mean weight.

    Edge weights are ``sim - sim.min()`` so they are non-negative. Clusters smaller than
    ``min_size`` are left ``UNASSIGNED``.
    """
    import igraph as ig
    import leidenalg as la

    edges, raw = mutual_knn_graph(sim, k)
    w = [x - float(sim.min()) for x in raw]
    graph = ig.Graph(n=len(sim), edges=edges)
    graph.es['weight'] = w
    gamma = float(np.mean(w)) * fraction if w else 0.0
    found = la.find_partition(
        graph,
        la.CPMVertexPartition,
        weights='weight',
        resolution_parameter=gamma,
        seed=seed,
        n_iterations=3,
    )
    part = np.array(found.membership, dtype=np.int64)
    small = np.bincount(part)[part] < min_size
    part[small] = UNASSIGNED
    return part


LEIDEN_FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def leiden_curve(
    sim: Floats,
    truth: Labels,
    labelled: Ints,
    k: int = 30,
    fractions: Sequence[float] = LEIDEN_FRACTIONS,
) -> list[CurvePoint]:
    """Pair precision/recall of Leiden CPM at each resolution fraction."""
    return [
        (float(f), *score_partition(leiden_cpm_partition(sim, k, f), truth, labelled)[1:])
        for f in fractions
    ]


def best_precision_at(curve: Sequence[CurvePoint], min_recall: float) -> float | None:
    """Highest precision among curve points with recall >= ``min_recall`` (an oracle cut)."""
    ok = [p for _, p, r, _ in curve if p is not None and r is not None and r >= min_recall]
    return round(max(ok), 3) if ok else None


def best_precision_summary(curve: Sequence[CurvePoint]) -> dict[str, float | None]:
    return {f'best_P@R>={r}': best_precision_at(curve, r) for r in RECALL_FLOORS}
