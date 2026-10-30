"""Retrieval and clustering metrics of the standard-benchmark protocol (torch; GPU or CPU).

Recall@K: a query scores 1 if any of its K nearest other rows (cosine, ties by index) has the same
class. k-means is an own Lloyd implementation (greedy k-means++ start) because K reaches 11,316.
"""

from __future__ import annotations

import math

import numpy as np
import torch


CHUNK = 512
KMEANS_CHUNK = 8192


def recall_hits(
    emb: np.ndarray,
    labels: np.ndarray,
    ks: tuple[int, ...],
    device: str = 'cpu',
    copies: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean ``(n, len(ks))``: query ``i`` has a same-class row among its ``ks[j]`` nearest others.

    ``copies`` (optional) gives each row a group id; rows of the query's own group are never its
    neighbours, which removes byte-identical files from the gallery (a diagnostic, not the protocol).
    """
    x = torch.as_tensor(emb, dtype=torch.float32, device=device)
    y = torch.as_tensor(np.unique(labels, return_inverse=True)[1], device=device)
    n, kmax = len(x), max(ks)
    if kmax >= n:
        raise ValueError(f'max K {kmax} needs more than {n} rows')
    g = (
        None
        if copies is None
        else torch.as_tensor(np.unique(copies, return_inverse=True)[1], device=device)
    )
    out = torch.zeros((n, len(ks)), dtype=torch.bool, device=device)
    for s in range(0, n, CHUNK):
        sim = x[s : s + CHUNK] @ x.T
        sim[
            torch.arange(sim.shape[0], device=device),
            torch.arange(s, s + sim.shape[0], device=device),
        ] = -torch.inf
        if g is not None:
            sim[g[s : s + CHUNK, None] == g[None, :]] = -torch.inf
        order = torch.sort(sim, dim=1, descending=True, stable=True).indices[:, :kmax]
        same = y[order] == y[s : s + CHUNK, None]
        for j, k in enumerate(ks):
            out[s : s + CHUNK, j] = same[:, :k].any(1)
    return np.asarray(out.cpu().numpy())


def top1_similarity(emb: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Cosine similarity of each row to its nearest other row (a near-duplicate diagnostic)."""
    x = torch.as_tensor(emb, dtype=torch.float32, device=device)
    out = []
    for s in range(0, len(x), CHUNK):
        sim = x[s : s + CHUNK] @ x.T
        sim[
            torch.arange(sim.shape[0], device=device),
            torch.arange(s, s + sim.shape[0], device=device),
        ] = -torch.inf
        out.append(sim.max(1).values.cpu())
    return np.asarray(torch.cat(out).numpy())


def kmeans(
    emb: np.ndarray,
    k: int,
    seed: int,
    device: str = 'cpu',
    max_iter: int = 300,
    tol: float = 1e-4,
) -> np.ndarray:
    """Lloyd k-means with k-means++ seeding; an empty cluster keeps its centre. Returns assignments."""
    x = torch.as_tensor(emb, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    centres = _kmeanspp(x, k, gen)
    prev = float('inf')
    assign = torch.zeros(len(x), dtype=torch.long, device=device)
    for _ in range(max_iter):
        assign, inertia = _assign(x, centres)
        sums = torch.zeros_like(centres).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).float()
        full = counts > 0
        centres[full] = sums[full] / counts[full, None]
        if prev - inertia <= tol * max(inertia, 1e-12):
            break
        prev = inertia
    assign, _ = _assign(x, centres)
    return np.asarray(assign.cpu().numpy())


def _kmeanspp(x: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """Greedy k-means++ (scikit-learn's variant): of ``2 + log k`` D^2-sampled candidates keep the one that lowers the potential most."""
    n = len(x)
    trials = 2 + int(math.log(k))
    first = int(torch.randint(n, (1,), generator=gen, device=x.device))
    idx = [first]
    xn = (x**2).sum(1)
    best = (xn - 2 * x @ x[first] + xn[first]).clamp_min(0)
    for _ in range(k - 1):
        total = best.sum()
        probs = best / total if total > 0 else torch.full_like(best, 1.0 / n)
        cand = torch.multinomial(probs, trials, replacement=True, generator=gen)
        d2 = (xn[:, None] - 2 * x @ x[cand].T + xn[cand][None, :]).clamp_min(0)
        merged = torch.minimum(best[:, None], d2)
        win = int(merged.sum(0).argmin())
        idx.append(int(cand[win]))
        best = merged[:, win]
    return x[torch.tensor(idx, device=x.device)].clone()


def _assign(x: torch.Tensor, centres: torch.Tensor) -> tuple[torch.Tensor, float]:
    cn = (centres**2).sum(1)
    parts, inertia = [], 0.0
    for s in range(0, len(x), KMEANS_CHUNK):
        xb = x[s : s + KMEANS_CHUNK]
        d2 = (xb**2).sum(1, keepdim=True) - 2 * xb @ centres.T + cn[None, :]
        m, a = d2.min(1)
        parts.append(a)
        inertia += float(m.clamp_min(0).sum())
    return torch.cat(parts), inertia


def nmi(truth: np.ndarray, cluster: np.ndarray) -> float:
    """Normalised mutual information, arithmetic normalisation (scikit-learn's default)."""
    t = np.unique(truth, return_inverse=True)[1]
    c = np.unique(cluster, return_inverse=True)[1]
    n = len(t)
    cell, cnt = np.unique(t.astype(np.int64) * (c.max() + 1) + c, return_counts=True)
    ti, ci = cell // (c.max() + 1), cell % (c.max() + 1)
    pt, pc = np.bincount(t) / n, np.bincount(c) / n
    pj = cnt / n
    mi = float((pj * np.log(pj / (pt[ti] * pc[ci]))).sum())
    ht = float(-(pt * np.log(pt)).sum())
    hc = float(-(pc * np.log(pc)).sum())
    mean = (ht + hc) / 2
    return 1.0 if mean == 0 else mi / mean


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Mean of ``a - b`` per query and its 95 % percentile interval over query resamples."""
    d = a.astype(np.float64) - b.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.array(
        [d[rng.integers(0, len(d), len(d))].mean() for _ in range(resamples)], dtype=np.float64
    )
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'delta': float(d.mean()), 'lo': float(lo), 'hi': float(hi)}
