"""FG-6: can a queue of crops of NEW models be built, on the class-disjoint split of set C.

In each make with two or more models, half the models are held out as new (``setc.split_new_models``).
A crop's novelty is 1 minus its mean cosine to its 10 nearest known examples of its own make: the
calibration-half crops of the make's known models (other photos only). AUROC separates new-model
crops from known-model crops of the same make on the scored half. The flag threshold is a quantile
of the known examples' own novelty, the lowest quantile keeping flag precision at or above the bar
on the calibration half, then read once on the other half. Same recipe as the class-level run
(``cluster_diag/novelty.py``), with the make as the key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from scripts.research.fine_grained.data import Universe

K = 10
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
PRECISION_BAR = 0.80
MIN_EACH = 5  # new-model and known crops a make needs on a half before its AUROC is read


def auroc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Probability that a random positive outscores a random negative (ties count half)."""
    if not len(positive) or not len(negative):
        return None
    diff = positive[:, None] - negative[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def novelty(
    vec: np.ndarray, rows: np.ndarray, examples: np.ndarray, groups: np.ndarray, k: int = K
) -> np.ndarray:
    """1 - mean cosine of each row to its k nearest example rows from a different group."""
    sim = vec[rows] @ vec[examples].T
    sim[groups[rows][:, None] == groups[examples][None, :]] = -np.inf
    top = np.sort(sim, axis=1)[:, -k:]
    top = np.where(np.isfinite(top), top, np.nan)
    return np.asarray(1.0 - np.nanmean(top, axis=1))


def _photo_group(u: Universe) -> np.ndarray:
    """Set C has no photo ids: the file's directory is its generation folder, the file its photo."""
    return np.array([m['id'] for m in u.meta])


def _score_make(
    vec: np.ndarray, u: Universe, make: str, cal: np.ndarray, test: np.ndarray
) -> dict[str, Any] | None:
    groups = _photo_group(u)
    is_make = u.make == make
    known_cal = cal[is_make[cal] & ~u.new_model[cal]]
    if len(known_cal) <= K:
        return None
    # own-class threshold candidates: novelty of the known examples against the other examples
    own = novelty(vec, known_cal, known_cal, groups)

    def side(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = rows[is_make[rows]]
        s = novelty(vec, rows, known_cal, groups)
        return s[u.new_model[rows]], s[~u.new_model[rows]]

    pos_cal, neg_cal = side(cal)
    pos_test, neg_test = side(test)
    if min(len(pos_test), len(neg_test), len(pos_cal), len(neg_cal)) < MIN_EACH:
        return None
    chosen = None
    for q in QUANTILES:
        thr = float(np.quantile(own, q))
        flagged = int((pos_cal > thr).sum() + (neg_cal > thr).sum())
        prec = (pos_cal > thr).sum() / flagged if flagged else 0.0
        if prec >= PRECISION_BAR:
            chosen = (q, thr)
            break
    out: dict[str, Any] = {
        'auroc': auroc(pos_test, neg_test),
        'new_crops': len(pos_test),
        'known_crops': len(neg_test),
        'chosen_quantile': None if chosen is None else chosen[0],
    }
    if chosen is not None:
        thr = chosen[1]
        flagged = int((pos_test > thr).sum() + (neg_test > thr).sum())
        out['flagged'] = flagged
        out['flag_precision'] = float((pos_test > thr).sum() / flagged) if flagged else None
        out['new_recall'] = float((pos_test > thr).mean())
    return out


def run_fold(vec: np.ndarray, u: Universe, fold: int) -> dict[str, Any]:
    cal, test = u.fold(fold)
    makes = sorted({m for m, new in zip(u.make, u.new_model, strict=True) if new and m})
    per_make = {m: _score_make(vec, u, m, cal, test) for m in makes}
    read = {m: v for m, v in per_make.items() if v is not None}
    flagged = sum(v.get('flagged', 0) for v in read.values())
    true_new = sum(
        round(v['flag_precision'] * v['flagged']) for v in read.values() if v.get('flagged')
    )
    return {
        'per_make': per_make,
        'makes_read': len(read),
        'makes_with_new_models': len(makes),
        'pooled_flag_precision': (true_new / flagged) if flagged else None,
        'auroc_by_make': {m: v['auroc'] for m, v in read.items()},
    }
