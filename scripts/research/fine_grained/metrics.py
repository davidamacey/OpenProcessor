"""Fine-grained metrics: ranking scores at three label levels, partition scores, review cost.

Pure numpy over item-aligned arrays. Pair and BCubed scores come from the evaluation harness
(``scripts/bench/cluster_eval_metrics.py``) so a figure here means what it means there.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench.cluster_eval_metrics import UNASSIGNED, bcubed_scores, pairwise_scores


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]

SHEET_CELLS = 24
TARGET_RECALLS = (0.05, 0.1, 0.2, 0.3)
MIN_PRECISION = 0.90


def _rank_order(sim: Floats) -> Ints:
    """Neighbours of each row nearest first, the row itself removed; ties by index."""
    s = np.array(sim, dtype=np.float64)
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1, kind='stable')[:, : len(s) - 1]


def ranking_scores(sim: Floats, truth: Sequence[Any] | Ints, k: int = 10) -> dict[str, float]:
    """P@1, P@k, mAP@R and k-NN vote accuracy among the rows (queries with no same-class peer skipped)."""
    y = np.asarray(truth)
    order = _rank_order(sim)
    hit = y[order] == y[:, None]
    n_same = hit.sum(1)
    use = n_same > 0
    p1 = float(hit[use, 0].mean())
    pk = float(hit[use, :k].mean())
    aps, votes = [], []
    for i in np.nonzero(use)[0]:
        r = int(n_same[i])
        h = hit[i, :r]
        aps.append(float((np.cumsum(h) / np.arange(1, r + 1) * h).sum() / r))
        top = y[order[i, :k]].tolist()
        counts = {c: top.count(c) for c in dict.fromkeys(top)}
        best = max(counts.values())
        votes.append(next(c for c in top if counts[c] == best) == y[i])
    return {
        'P@1': p1,
        f'P@{k}': pk,
        'mAP@R': float(np.mean(aps)),
        f'{k}NN_vote': float(np.mean(votes)),
        'queries': int(use.sum()),
    }


def partition_scores(part: Ints, model: Sequence[Any], make: Sequence[Any]) -> dict[str, Any]:
    """BCubed at model and make level, pair recall at model level and the within-make merge share.

    ``part`` holds one code per row; UNASSIGNED rows are singletons.
    """
    p = np.asarray(part)
    m, k = np.asarray(model), np.asarray(make)
    b_model, b_make = bcubed_scores(m, p), bcubed_scores(k, p)
    pair = pairwise_scores(m, p)
    return {
        'model': b_model,
        'make': b_make,
        'model_pair_recall': pair['pair_recall'],
        'model_pair_precision': pair['pair_precision'],
        'within_make_false_merge_share': within_make_false_merge_share(p, m, k),
        'clusters': len(set(p[p != UNASSIGNED].tolist())),
    }


def within_make_false_merge_share(
    part: Ints, model: Sequence[Any] | Ints, make: Sequence[Any] | Ints
) -> float | None:
    """Of the pairs sharing a cluster but not a model, the share that still share a make."""
    p, m, k = np.asarray(part), np.asarray(model), np.asarray(make)
    wrong = same_make = 0
    for code in set(p[p != UNASSIGNED].tolist()):
        idx = np.nonzero(p == code)[0]
        mm, kk = m[idx], k[idx]
        diff_model = mm[:, None] != mm[None, :]
        iu = np.triu_indices(len(idx), 1)
        d = diff_model[iu]
        wrong += int(d.sum())
        same_make += int((d & (kk[:, None] == kk[None, :])[iu]).sum())
    return same_make / wrong if wrong else None


def review_cost_r1(part: Ints, model: Sequence[Any]) -> float:
    """Decisions per labelled crop for a person naming groups from 6 x 4 sheets (plan section 4).

    A cluster of n crops costs ceil(n / 24) sheet decisions plus one per crop that is not of its
    plurality model (the oracle flagger); a singleton (or UNASSIGNED row) costs one.
    """
    p, m = np.asarray(part), np.asarray(model)
    decisions = int((p == UNASSIGNED).sum())
    for code in set(p[p != UNASSIGNED].tolist()):
        members = m[p == code]
        if len(members) == 1:
            decisions += 1
            continue
        _, counts = np.unique(members, return_counts=True)
        decisions += math.ceil(len(members) / SHEET_CELLS) + int(len(members) - counts.max())
    return decisions / len(p)


def choose_operating_point(
    points: Sequence[Mapping[str, float]],
    targets: Sequence[float] = TARGET_RECALLS,
    min_precision: float = MIN_PRECISION,
) -> dict[str, Any] | None:
    """Plan section 6's rule at model level, on calibration points ``{param, recall, precision}``.

    For each target recall T, the candidate is the most precise point whose pair recall is at least
    T (ties: the higher recall). It qualifies when its BCubed precision is at least
    ``min_precision``. The answer is the qualifying point of the largest T; None when none qualifies.
    """
    chosen = None
    for t in sorted(targets):
        ok = [p for p in points if p['recall'] >= t]
        if not ok:
            continue
        best = max(ok, key=lambda p: (p['precision'], p['recall']))
        if best['precision'] >= min_precision:
            chosen = {**best, 'target': t}
    return chosen


def matched_point(
    curve: Sequence[Mapping[str, float]], recall: float, tol: float = 0.02
) -> Mapping[str, float] | None:
    """The curve point of highest BCubed precision among those whose recall is within ``tol`` of ``recall``."""
    near = [p for p in curve if abs(p['recall'] - recall) <= tol]
    return max(near, key=lambda p: p['precision']) if near else None
