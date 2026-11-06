"""Grouping rules and metrics for the near-duplicate prototypes.

Edges are exact (blocked matmul over unit-norm rows). Three rules, each returning ``labels`` (one group id
per row, singletons included):

* ``single``: connected components, today's rule (``frame_dedup.near_dup_groups`` computes the same
  components; :func:`product_groups` runs the product function so the control rows are its numbers);
* ``single_guard``: components larger than ``guard`` are re-split by complete linkage at the same cut;
* ``complete``: complete linkage inside every component.

``truth`` is the family id (the original an item derives from); a *copy* is recovered when it shares a
group with the original of its family.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform


MIN_EDGE_COS = 0.85


@dataclass(frozen=True)
class Edges:
    """Pairs ``i < j`` with cosine ``>= floor``."""

    i: np.ndarray
    j: np.ndarray
    cos: np.ndarray
    n: int

    def at(self, cut: float) -> tuple[np.ndarray, np.ndarray]:
        keep = self.cos >= cut
        return self.i[keep], self.j[keep]


def edges_from_vectors(x: np.ndarray, floor: float = MIN_EDGE_COS, block: int = 2048) -> Edges:
    n = x.shape[0]
    xi, xj, xc = [], [], []
    xt = np.ascontiguousarray(x.T)
    for r0 in range(0, n, block):
        sims = x[r0 : r0 + block] @ xt
        rows, cols = np.nonzero(sims >= floor)
        gi = rows + r0
        keep = cols > gi
        xi.append(gi[keep])
        xj.append(cols[keep])
        xc.append(sims[rows[keep], cols[keep]])
    cat = np.concatenate
    return Edges(
        cat(xi).astype(np.int64) if xi else np.zeros(0, np.int64),
        cat(xj).astype(np.int64) if xj else np.zeros(0, np.int64),
        cat(xc).astype(np.float32) if xc else np.zeros(0, np.float32),
        n,
    )


def components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    adj = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    return np.asarray(connected_components(adj, directed=False)[1], dtype=np.int64)


def _complete_split(x: np.ndarray, members: np.ndarray, cut: float) -> np.ndarray:
    """Complete-linkage labels (0..k-1) of ``members`` at cosine ``cut``."""
    sub = x[members]
    dist = np.clip(1.0 - sub @ sub.T, 0.0, None).astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform((dist + dist.T) / 2.0, checks=False), method='complete')
    return np.asarray(fcluster(tree, t=1.0 - cut, criterion='distance'), dtype=np.int64) - 1


def group_labels(
    x: np.ndarray, edges: Edges, cut: float, rule: str, *, guard: int = 50
) -> tuple[np.ndarray, int]:
    """``(labels, n_split)`` for ``rule``; ``n_split`` counts components re-split by complete linkage."""
    comp = components(edges.n, *edges.at(cut))
    if rule == 'single':
        return comp, 0
    sizes = np.bincount(comp)
    todo = np.flatnonzero(sizes >= 2) if rule == 'complete' else np.flatnonzero(sizes > guard)
    if rule not in {'complete', 'single_guard'}:
        raise ValueError(f'unknown rule {rule!r}')
    labels = comp.copy()
    next_id = int(comp.max()) + 1 if len(comp) else 0
    n_split = 0
    for c in todo:
        members = np.flatnonzero(comp == c)
        sub = _complete_split(x, members, cut)
        if sub.max() > 0:
            n_split += 1
        labels[members] = next_id + sub
        next_id += int(sub.max()) + 1
    return labels, n_split


def product_groups(x: np.ndarray, cut: float) -> list[list[int]]:
    """Today's product function, unchanged, on the same vectors (the control rows)."""
    from src.services.detection.frame_dedup import near_dup_groups

    return near_dup_groups(x, threshold=cut)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % interval for ``k`` successes of ``n``."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate(
    labels: np.ndarray, family: np.ndarray, transform: np.ndarray, original_index: dict[int, int]
) -> dict[str, object]:
    """Copy recall per transform, false-group rate, BCubed and group sizes.

    ``original_index[family]`` is the row of the original of each family; ``transform == 'orig'`` marks it.
    """
    copy_rows = np.flatnonzero(transform != 'orig')
    orig_rows = np.array([original_index[int(family[r])] for r in copy_rows], dtype=np.int64)
    hit = labels[copy_rows] == labels[orig_rows]
    per: dict[str, dict[str, object]] = {}
    for t in sorted(set(transform[copy_rows])):
        sel = transform[copy_rows] == t
        k, n = int(hit[sel].sum()), int(sel.sum())
        per[str(t)] = {'recovered': k, 'n': n, 'recall': k / n if n else None, 'ci': wilson(k, n)}
    k_all, n_all = int(hit.sum()), len(hit)
    sizes = np.bincount(labels)
    multi = np.flatnonzero(sizes >= 2)
    fam_per_group: dict[int, set[int]] = {}
    for row in np.flatnonzero(np.isin(labels, multi)):
        fam_per_group.setdefault(int(labels[row]), set()).add(int(family[row]))
    mixed = sum(1 for s in fam_per_group.values() if len(s) > 1)
    return {
        'per_transform': per,
        'pooled_recall': {
            'recovered': k_all,
            'n': n_all,
            'recall': k_all / max(n_all, 1),
            'ci': wilson(k_all, n_all),
        },
        'groups': len(multi),
        'false_groups': mixed,
        'false_group_rate': mixed / max(len(multi), 1),
        'false_group_ci': wilson(mixed, len(multi)),
        'largest_group': int(sizes.max()) if len(sizes) else 0,
        'bcubed': bcubed(labels, family),
    }


def bcubed(labels: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """BCubed precision and recall of ``labels`` against ``truth``."""
    n = len(labels)
    if n == 0:
        return {'precision': 1.0, 'recall': 1.0}
    pair = np.stack([labels, truth], axis=1)
    _, joint = np.unique(pair, axis=0, return_counts=True)
    n_lab = np.bincount(labels)
    _, truth_inv = np.unique(truth, return_inverse=True)
    n_truth = np.bincount(truth_inv)
    _, pair_inv = np.unique(pair, axis=0, return_inverse=True)
    c = joint[pair_inv.reshape(-1)]
    return {
        'precision': float(np.mean(c / n_lab[labels])),
        'recall': float(np.mean(c / n_truth[truth_inv])),
    }
