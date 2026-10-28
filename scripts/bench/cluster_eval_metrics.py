"""Pure cluster-partition metrics against independent ground-truth classes.

A partition is one cluster code per item, ``UNASSIGNED`` (-1) for an item no cluster claimed.
Ground truth is one class code per item. Everything is numpy over item-aligned arrays, and pair
counts come from a sparse cluster-by-class table, so 100k items cost milliseconds.

For curation a wrong propagated label costs more than an item sent to review, so false merges
(different-class pairs sharing a cluster) and false splits (same-class pairs kept apart) are
reported as separate rates, never folded into one F-score. An unassigned item is a missed pair,
not a merged one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.bench.parity_metrics import adjusted_rand_index


UNASSIGNED = -1


def _pairs(counts: np.ndarray) -> int:
    c = counts.astype(np.int64)
    return int((c * (c - 1) // 2).sum())


def _codes(values: np.ndarray) -> np.ndarray:
    return np.unique(values, return_inverse=True)[1].astype(np.int64)


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def pairwise_scores(truth: np.ndarray, cluster: np.ndarray) -> dict[str, Any]:
    """Pair-counting precision, recall and the separate false-merge and false-split rates."""
    n = len(truth)
    t_all = _codes(truth)
    assigned = cluster != UNASSIGNED
    t, c = t_all[assigned], _codes(cluster[assigned])
    n_classes = int(t_all.max()) + 1 if n else 0
    cell, cell_n = np.unique(c * max(n_classes, 1) + t, return_counts=True)
    cell_cluster = cell // max(n_classes, 1)
    tp = _pairs(cell_n)
    same_cluster = _pairs(np.bincount(cell_cluster, weights=cell_n).astype(np.int64))
    same_class = _pairs(np.bincount(t_all))
    total = n * (n - 1) // 2
    fp, fn = same_cluster - tp, same_class - tp
    # share of assigned items that sit outside their cluster's majority class
    majority = np.zeros(int(c.max()) + 1 if len(c) else 0, dtype=np.int64)
    np.maximum.at(majority, cell_cluster, cell_n)
    mixed = int((np.bincount(cell_cluster, minlength=len(majority)) > 1).sum())
    n_assigned = int(assigned.sum())
    precision, recall = _rate(tp, same_cluster), _rate(tp, same_class)
    return {
        'items': n,
        'assigned': n_assigned,
        'unassigned_fraction': _rate(n - n_assigned, n),
        'clusters': len(majority),
        'mixed_clusters': mixed,
        'pair_precision': precision,
        'pair_recall': recall,
        'false_merge_rate': _rate(fp, same_cluster),
        'false_split_rate': _rate(fn, same_class),
        'correct_split_recovery': None if total == same_class else 1 - fp / (total - same_class),
        'minority_item_rate': _rate(n_assigned - int(majority.sum()), n_assigned),
    }


def stratified_scores(
    truth: np.ndarray, cluster: np.ndarray, strata: np.ndarray
) -> dict[str, dict[str, Any]]:
    """``pairwise_scores`` of the items of each stratum, clusters taken as they are."""
    return {
        str(s): pairwise_scores(truth[strata == s], cluster[strata == s]) for s in np.unique(strata)
    }


def assignment_stability(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Agreement of two runs: ARI over the items both assigned, plus exact-grouping identity."""
    both = (first != UNASSIGNED) & (second != UNASSIGNED)
    ari = adjusted_rand_index(first[both], second[both]) if both.any() else None
    same_holes = bool(((first == UNASSIGNED) == (second == UNASSIGNED)).all())
    return {
        'compared_items': int(both.sum()),
        'ari': ari,
        'identical_grouping': same_holes and (ari == 1.0 or not both.any()),
    }


def moved_items(reference: np.ndarray, candidate: np.ndarray, protected: np.ndarray) -> int:
    """Protected items whose cluster changes from ``reference`` to ``candidate``.

    Cluster ids are arbitrary names, so each reference cluster is mapped to the candidate cluster
    holding most of its assigned members; a protected item is moved when it is not in that
    cluster, including when it lost its cluster or gained one it did not have.
    """
    kept = (reference != UNASSIGNED) & (candidate != UNASSIGNED)
    pair, count = np.unique(
        np.stack([reference[kept], candidate[kept]]), axis=1, return_counts=True
    )
    order = np.lexsort((-count, pair[0]))  # per reference cluster, largest overlap first
    lead = order[np.r_[True, pair[0][order][1:] != pair[0][order][:-1]]]
    target = dict(zip(pair[0][lead].tolist(), pair[1][lead].tolist(), strict=True))
    lost = UNASSIGNED - 1  # a reference cluster with no assigned candidate member matches nothing
    expected = np.array(
        [
            UNASSIGNED if r == UNASSIGNED else target.get(r, lost)
            for r in reference[protected].tolist()
        ],
        dtype=np.int64,
    )
    return int((candidate[protected] != expected).sum())
