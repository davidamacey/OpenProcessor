"""Statistics for judge agreement: intervals and kappas. Pure, no I/O."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0, 1) when there is no sample."""
    if successes > n or successes < 0:
        raise ValueError(f'successes {successes} outside 0..{n}')
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cluster_bootstrap(
    values_by_sheet: Mapping[str, Sequence[float]],
    *,
    seed: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile interval of the pooled mean, resampling whole sheets (cells are correlated)."""
    sheets = [list(v) for _, v in sorted(values_by_sheet.items()) if len(v)]
    if not sheets:
        return None
    rng = random.Random(seed)
    sums = [sum(v) for v in sheets]
    counts = [len(v) for v in sheets]
    means: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(sheets)) for _ in sheets]
        means.append(sum(sums[i] for i in idx) / sum(counts[i] for i in idx))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa of two raters; None when chance agreement is already 1 (no variation)."""
    if len(a) != len(b):
        raise ValueError('ratings must have the same length')
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    chance = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if math.isclose(chance, 1.0):
        return None
    return (observed - chance) / (1 - chance)


def fleiss_kappa(ratings: Sequence[Sequence[Hashable]]) -> float | None:
    """Fleiss' kappa: one list of rater labels per subject, the same number of raters each."""
    if not ratings:
        return None
    raters = len(ratings[0])
    if raters < 2 or any(len(r) != raters for r in ratings):
        raise ValueError('every subject needs the same number (at least 2) of raters')
    n = len(ratings)
    category_totals: Counter[Hashable] = Counter()
    agreement = 0.0
    for subject in ratings:
        counts = Counter(subject)
        category_totals.update(counts)
        agreement += (sum(c * c for c in counts.values()) - raters) / (raters * (raters - 1))
    observed = agreement / n
    chance = sum((t / (n * raters)) ** 2 for t in category_totals.values())
    if math.isclose(chance, 1.0):
        return None
    return (observed - chance) / (1 - chance)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def binary_scores(truth: Sequence[bool], pred: Sequence[bool]) -> dict[str, Any]:
    """Accuracy, kappa, precision/recall of the positive class and both error rates."""
    if len(truth) != len(pred):
        raise ValueError('truth and prediction must have the same length')
    tp = sum(t and p for t, p in zip(truth, pred, strict=True))
    fp = sum((not t) and p for t, p in zip(truth, pred, strict=True))
    fn = sum(t and (not p) for t, p in zip(truth, pred, strict=True))
    tn = len(truth) - tp - fp - fn
    return {
        'n': len(truth),
        'accuracy': _ratio(tp + tn, len(truth)),
        'kappa': cohens_kappa(list(truth), list(pred)),
        'precision': _ratio(tp, tp + fp),
        'recall': _ratio(tp, tp + fn),
        'false_negative_rate': _ratio(fn, tp + fn),
        'false_positive_rate': _ratio(fp, fp + tn),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }
