"""The standard 10-fold verification protocol: accuracy and TAR at a fixed FAR."""

from __future__ import annotations

from typing import Any

import numpy as np


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold (accept when score >= it) with the highest accuracy over the given pairs."""
    pos, neg = np.sort(scores[labels == 1]), np.sort(scores[labels == 0])
    cand = np.unique(scores)
    correct = (
        np.searchsorted(neg, cand, side='left') + len(pos) - np.searchsorted(pos, cand, side='left')
    )
    return float(cand[int(np.argmax(correct))])


def fold_accuracy(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    """Per fold: pick the accuracy-maximising threshold on the other folds, score the held-out fold."""
    accs = []
    for f in np.unique(folds):
        test = folds == f
        thr = best_threshold(scores[~test], labels[~test])
        accs.append(float(((scores[test] >= thr) == (labels[test] == 1)).mean()))
    return {'accuracy': float(np.mean(accs)), 'std': float(np.std(accs))}


def tar_at_far(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray, far: float) -> float:
    """Mean over folds of the true accept rate at the threshold that gives ``far`` on the train folds."""
    tars = []
    for f in np.unique(folds):
        test = folds == f
        neg = np.sort(scores[~test][labels[~test] == 0])[::-1]
        thr = neg[min(int(np.floor(far * len(neg))), len(neg) - 1)]
        pos = scores[test][labels[test] == 1]
        tars.append(float((pos > thr).mean()))
    return float(np.mean(tars))
