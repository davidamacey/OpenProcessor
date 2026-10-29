"""Held-out protocol: pick the cut on one half of the source images, score it on the other.

An oracle threshold chosen on the scored labels flatters every method, so each variant is
scored the way it would be used: its cut is the most precise one on the calibration half that
still reaches the target recall there, and the number reported is that cut on the other half.
Both directions (A to B and B to A) are run. Groups (source images) never straddle the halves.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench.cluster_eval_metrics import pairwise_scores


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray


def split_groups(groups: Iterable[str], seed: int) -> tuple[list[str], list[str]]:
    """Two halves of the distinct groups, ranked by sha256(seed:group); the odd one goes second.

    The ranking depends only on the seed and the group name, so the split is the same for any
    input order and any other group present.
    """
    ranked = sorted(set(groups), key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    half = len(ranked) // 2
    return sorted(ranked[:half]), sorted(ranked[half:])


def half_mask(item_groups: Sequence[str], first_half: Iterable[str]) -> NDArray[np.bool_]:
    """True for items whose group is in ``first_half``."""
    members = set(first_half)
    return np.array([g in members for g in item_groups], dtype=bool)


def score_on(
    partition: NDArray[np.integer[Any]], truth: NDArray[Any], mask: NDArray[np.bool_]
) -> tuple[float | None, float | None, float]:
    """Pair precision, recall and unassigned share of the masked labelled items."""
    s = pairwise_scores(truth[mask], partition[mask])
    return s['pair_precision'], s['pair_recall'], float(s['unassigned_fraction'] or 0.0)


def choose_and_score(
    candidates: Sequence[tuple[Any, NDArray[np.integer[Any]]]],
    truth: NDArray[Any],
    calibration: NDArray[np.bool_],
    heldout: NDArray[np.bool_],
    min_recall: float,
) -> dict[str, Any] | None:
    """Best calibration-half cut at ``min_recall`` and its score on the held-out half.

    ``candidates`` are ``(parameter, partition)`` pairs with the partition aligned to
    ``truth``. Returns ``None`` when no candidate reaches the recall on the calibration half.
    """
    best: tuple[float, Any, NDArray[np.integer[Any]], float] | None = None
    for param, part in candidates:
        precision, recall, _ = score_on(part, truth, calibration)
        if precision is None or recall is None or recall < min_recall or math.isnan(precision):
            continue
        if best is None or precision > best[0]:
            best = (precision, param, part, recall)
    if best is None:
        return None
    precision, param, part, recall = best
    t_precision, t_recall, t_unassigned = score_on(part, truth, heldout)
    return {
        'parameter': param,
        'calibration': {'precision': precision, 'recall': recall},
        'heldout': {
            'precision': t_precision,
            'recall': t_recall,
            'unassigned': t_unassigned,
        },
    }


def two_fold(
    candidates: Sequence[tuple[Any, NDArray[np.integer[Any]]]],
    truth: NDArray[Any],
    first_half: NDArray[np.bool_],
    min_recall: float,
) -> list[dict[str, Any] | None]:
    """``choose_and_score`` with the halves in both roles."""
    second_half = ~first_half
    return [
        choose_and_score(candidates, truth, first_half, second_half, min_recall),
        choose_and_score(candidates, truth, second_half, first_half, min_recall),
    ]
