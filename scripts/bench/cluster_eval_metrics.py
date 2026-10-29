"""Pure cluster-partition metrics against independent ground-truth classes.

A partition is one cluster code per item, ``UNASSIGNED`` (-1) for an item no cluster claimed.
Ground truth is one class code per item. Everything is numpy over item-aligned arrays, and pair
counts come from a sparse cluster-by-class table, so 100k items cost milliseconds.

For curation a wrong propagated label costs more than an item sent to review, so false merges
(different-class pairs sharing a cluster) and false splits (same-class pairs kept apart) are
reported as separate rates, never folded into one F-score. An unassigned item is a missed pair,
not a merged one.

BCubed and AMI have no notion of "no cluster", so there an unassigned (abstained) item counts as
its own singleton cluster; ``unassigned_fraction`` and ``risk_coverage`` report the abstention
itself next to them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

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


def _count_pairs(*keys: np.ndarray) -> int:
    """Pairs of items that share every one of ``keys`` (parallel arrays over the same items)."""
    if not len(keys[0]):
        return 0
    return _pairs(np.unique(np.stack(keys), axis=1, return_counts=True)[1])


def _with_singletons(cluster: np.ndarray) -> np.ndarray:
    """Cluster codes where each unassigned item gets a code of its own."""
    out = _codes(cluster)
    hole = cluster == UNASSIGNED
    out[hole] = out.max(initial=-1) + 1 + np.arange(int(hole.sum()))
    return out


def bcubed_scores(truth: np.ndarray, cluster: np.ndarray) -> dict[str, Any]:
    """BCubed precision, recall and F (Amigo et al. 2009); unassigned items are singletons."""
    n = len(truth)
    if not n:
        return {'bcubed_precision': None, 'bcubed_recall': None, 'bcubed_f': None}
    t, c = _codes(truth), _with_singletons(cluster)
    n_t = np.bincount(t)
    n_c = np.bincount(c)
    cell, cell_n = np.unique(c * len(n_t) + t, return_counts=True)
    cell_c, cell_t = cell // len(n_t), cell % len(n_t)
    precision = float((cell_n * cell_n / n_c[cell_c]).sum()) / n
    recall = float((cell_n * cell_n / n_t[cell_t]).sum()) / n
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'bcubed_precision': precision, 'bcubed_recall': recall, 'bcubed_f': f}


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def _expected_mutual_info(rows: np.ndarray, cols: np.ndarray, n: int) -> float:
    """Expected MI of a random contingency table with these marginals (hypergeometric model).

    Only distinct marginal values matter, so the sum runs over unique row and column totals,
    weighted by how many rows and columns carry them.
    """
    a_val, a_cnt = np.unique(rows, return_counts=True)
    b_val, b_cnt = np.unique(cols, return_counts=True)
    gl = gammaln(np.arange(n + 2, dtype=np.float64) + 1)  # gl[k] = log(k!)
    total = 0.0
    for a, wa in zip(a_val.tolist(), a_cnt.tolist(), strict=True):
        for b, wb in zip(b_val.tolist(), b_cnt.tolist(), strict=True):
            nij = np.arange(max(1, a + b - n), min(a, b) + 1)
            if not len(nij):
                continue
            log_p = (
                gl[a] + gl[b] + gl[n - a] + gl[n - b] - gl[n]
                - gl[nij] - gl[a - nij] - gl[b - nij] - gl[n - a - b + nij]
            )  # fmt: skip
            term = nij / n * (np.log(n * nij) - np.log(a) - np.log(b))
            total += wa * wb * float((term * np.exp(log_p)).sum())
    return total


def adjusted_mutual_info(truth: np.ndarray, cluster: np.ndarray) -> float | None:
    """AMI with the arithmetic-mean normaliser (scikit-learn's default); holes are singletons."""
    n = len(truth)
    if not n:
        return None
    t, c = _codes(truth), _with_singletons(cluster)
    rows, cols = np.bincount(t), np.bincount(c)
    if len(rows) == len(cols) == 1 or len(rows) == len(cols) == n:
        return 1.0
    cell, cell_n = np.unique(t * len(cols) + c, return_counts=True)
    ct, cc = cell // len(cols), cell % len(cols)
    mi = float((cell_n / n * (np.log(n * cell_n) - np.log(rows[ct]) - np.log(cols[cc]))).sum())
    emi = _expected_mutual_info(rows, cols, n)
    norm = (_entropy(rows) + _entropy(cols)) / 2
    denom = norm - emi
    denom = (
        min(denom, -np.finfo(np.float64).eps) if denom < 0 else max(denom, np.finfo(np.float64).eps)
    )
    return (mi - emi) / denom


def risk_coverage(
    truth: np.ndarray, cluster: np.ndarray, confidence: np.ndarray | None = None
) -> dict[str, Any]:
    """Risk (share of accepted items outside their cluster's majority class) against coverage.

    An unassigned item is an abstention: it lowers coverage and is never part of the risk. With
    per-item ``confidence`` the curve accepts assigned items from most to least confident and
    ``aurc`` is the mean risk over those prefixes; without it there is the single operating point.
    """
    n = len(truth)
    assigned = np.flatnonzero(cluster != UNASSIGNED)
    t, c = _codes(truth)[assigned], _codes(cluster[assigned])
    wrong = np.zeros(len(assigned), dtype=bool)
    if len(assigned):
        cell, cell_n = np.unique(c * (int(t.max()) + 1) + t, return_counts=True)
        cell_c = cell // (int(t.max()) + 1)
        order = np.lexsort((cell, -cell_n, cell_c))  # per cluster: biggest class first
        lead = order[np.r_[True, cell_c[order][1:] != cell_c[order][:-1]]]
        majority = np.zeros(int(c.max()) + 1, dtype=np.int64)
        majority[cell_c[lead]] = cell[lead] % (int(t.max()) + 1)
        wrong = t != majority[c]
    out: dict[str, Any] = {
        'items': n,
        'abstained': n - len(assigned),
        'coverage': _rate(len(assigned), n),
        'risk': _rate(int(wrong.sum()), len(assigned)),
        'curve': None,
        'aurc': None,
    }
    if confidence is not None and len(assigned):
        rank = np.argsort(-confidence[assigned], kind='stable')
        risk = np.cumsum(wrong[rank]) / np.arange(1, len(rank) + 1)
        out['curve'] = [(k / n, float(r)) for k, r in enumerate(risk, 1)]
        out['aurc'] = float(risk.mean())
    return out


def _half(same: bool, whole: int, same_attr: int) -> int:
    return same_attr if same else whole - same_attr


def attribute_pair_scores(
    truth: np.ndarray, cluster: np.ndarray, attribute: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Pair recall and false-merge rate for pairs that share (``same``) or differ in an attribute.

    The attribute is a view label, a crop-size bin or anything else per item; ``different`` is
    the hard stratum (a front photo against a rear photo of one model).
    """
    t, a = _codes(truth), _codes(attribute)
    kept = cluster != UNASSIGNED
    c_all = np.full(len(cluster), -1, dtype=np.int64)
    c_all[kept] = _codes(cluster[kept])
    tk, ak, ck = t[kept], a[kept], c_all[kept]
    totals = {
        'class': _count_pairs(t),
        'class_attr': _count_pairs(t, a),
        'cluster': _count_pairs(ck),
        'cluster_attr': _count_pairs(ck, ak),
        'tp': _count_pairs(ck, tk),
        'tp_attr': _count_pairs(ck, tk, ak),
    }
    out = {}
    for name, same in (('same', True), ('different', False)):
        same_class = _half(same, totals['class'], totals['class_attr'])
        co = _half(same, totals['cluster'], totals['cluster_attr'])
        tp = _half(same, totals['tp'], totals['tp_attr'])
        out[name] = {
            'same_class_pairs': same_class,
            'coclustered_pairs': co,
            'pair_recall': _rate(tp, same_class),
            'false_merge_rate': _rate(co - tp, co),
        }
    return out
