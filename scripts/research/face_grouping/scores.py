"""Identity-clustering scores for a face partition.

Pair and BCubed numbers come from ``scripts.bench.cluster_eval_metrics`` (nothing re-implemented); this
module adds the face-specific rates. An unassigned face (-1) counts as a singleton cluster everywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.bench.cluster_eval_metrics import bcubed_scores, pairwise_scores


PURITY = 0.99


def with_singletons(cluster: np.ndarray) -> np.ndarray:
    out = np.unique(cluster, return_inverse=True)[1].astype(np.int64)
    hole = cluster == -1
    out[hole] = out.max(initial=-1) + 1 + np.arange(int(hole.sum()))
    return out


def identity_report(truth: np.ndarray, cluster: np.ndarray, min_images: int = 10) -> dict[str, Any]:
    t = np.unique(truth, return_inverse=True)[1].astype(np.int64)
    c = with_singletons(cluster)
    n_t, n_c = int(t.max()) + 1, int(c.max()) + 1
    t_size, c_size = np.bincount(t, minlength=n_t), np.bincount(c, minlength=n_c)
    cell, cell_n = np.unique(c * n_t + t, return_counts=True)
    cell_c, cell_t = cell // n_t, cell % n_t

    multi = c_size[cell_c] >= 2
    strong = multi & (cell_n >= 2)
    per_cluster = np.bincount(cell_c[strong], minlength=n_c)
    big_clusters = int((c_size >= 2).sum())
    wrong = int((per_cluster >= 2).sum())

    spread = np.bincount(cell_t[multi], minlength=n_t)
    multi_ids = t_size >= 2

    pure = cell_n / c_size[cell_c] >= PURITY
    best = np.zeros(n_t)
    np.maximum.at(best, cell_t[pure], cell_n[pure])
    wanted = t_size >= min_images

    pw, bc = pairwise_scores(truth, c), bcubed_scores(truth, c)
    return {
        'items': len(t),
        'identities': n_t,
        'clusters': n_c,
        'singleton_share': float((c_size[c] == 1).mean()),
        'wrong_merge_clusters': wrong,
        'wrong_merge_rate': wrong / big_clusters if big_clusters else None,
        'split_rate': float((spread[multi_ids] > 1).mean()) if multi_ids.any() else None,
        'task_identities': int(wanted.sum()),
        'task_recall': float((best[wanted] / t_size[wanted]).mean()) if wanted.any() else None,
        'pair_precision': pw['pair_precision'],
        'pair_recall': pw['pair_recall'],
        'pair_f': _f(pw['pair_precision'], pw['pair_recall']),
        'bcubed_precision': bc['bcubed_precision'],
        'bcubed_recall': bc['bcubed_recall'],
        'bcubed_f': bc['bcubed_f'],
    }


def _f(p: float | None, r: float | None) -> float | None:
    return None if not p or not r else 2 * p * r / (p + r)


def best_point(rows: list[dict[str, Any]], precision_floor: float) -> dict[str, Any] | None:
    """The row with the highest BCubed recall among those at or above the BCubed precision floor."""
    ok = [r for r in rows if r['bcubed_precision'] >= precision_floor]
    return max(ok, key=lambda r: r['bcubed_recall']) if ok else None
