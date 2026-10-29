"""Does a cut chosen on one project carry to another, and how many examples does it take?

Transfer: the within-class cut chosen on the calibration half of the large public set is
applied unchanged to a second project's crops (same embedding model), and the second
project's held-out half is scored. Sample size: the calibration half is subsampled to a
fixed number of validated crops per detector class, the cut is re-chosen on the subsample,
and the spread of that cut around the full-calibration cut says how many examples a project
needs to calibrate its own cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench import scorecard
from scripts.research.cluster_diag import class_prior as cp, heldout as ho


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    Ints = NDArray[np.integer[Any]]
    Mask = NDArray[np.bool_]

PER_KEY_COUNTS = (5, 10, 20, 50)
TOTAL_COUNTS = (100, 200, 400, 800)
SEEDS = tuple(range(10))
TOLERANCE = 0.02
ROUNDING = 1e-9  # cut heights on a 0.005 grid differ by 0.02 up to float rounding
SEEDS_REQUIRED = 9
TRANSFER_BCUBED_PRECISION_BAR = 0.90


def subsample_per_key(keys: Any, eligible: Mask, per_key: int, seed: int) -> Mask:
    """A seeded mask with at most ``per_key`` of the eligible crops of each key."""
    k = np.asarray(keys)
    rng = np.random.default_rng(seed)
    chosen = np.zeros(len(k), dtype=bool)
    for name in sorted(set(k[eligible].tolist())):
        rows = np.nonzero(eligible & (k == name))[0]
        take = rng.permutation(rows)[:per_key]
        chosen[take] = True
    return chosen


def subsample_total(eligible: Mask, total: int, seed: int) -> Mask:
    """A seeded mask of ``total`` eligible crops drawn at random, keeping the class mix."""
    rows = np.nonzero(eligible)[0]
    chosen = np.zeros(len(eligible), dtype=bool)
    chosen[np.random.default_rng(seed).permutation(rows)[:total]] = True
    return chosen


def cut_from_sample(
    cands: Sequence[tuple[float, Ints]], truth: Any, sample: Mask, scored: Mask, target: float
) -> float | None:
    """The cut chosen on the sampled crops at the target pair recall (None if unreachable)."""
    chosen = ho.choose_and_score(cands, truth, sample, scored, target)
    return None if chosen is None else float(chosen['parameter'])


def within_tolerance(cut: float | None, reference: float | None) -> bool:
    """True if both cuts exist and differ by at most the tolerance (float rounding allowed)."""
    return (
        cut is not None and reference is not None and abs(cut - reference) <= TOLERANCE + ROUNDING
    )


def sample_size_table(
    cands: Sequence[tuple[float, Ints]],
    truth: Any,
    first_half: Mask,
    target: float,
    sampler: Callable[[Mask, int, int], Mask],
    counts: Sequence[int],
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Per sample size: the cuts of seeded subsamples of the calibration half against the full cut.

    ``sampler(eligible, count, seed)`` draws the subsample (per key or from the whole half).
    """
    full = cut_from_sample(cands, truth, first_half, ~first_half, target)
    rows: dict[str, Any] = {}
    for n in counts:
        cuts = [
            cut_from_sample(cands, truth, sampler(first_half, n, s), ~first_half, target)
            for s in seeds
        ]
        near = [within_tolerance(c, full) for c in cuts]
        rows[str(n)] = {'cuts': cuts, 'within_tolerance': int(sum(near)), 'seeds': len(cuts)}
    enough = [int(n) for n, r in rows.items() if r['within_tolerance'] >= SEEDS_REQUIRED]
    return {
        'full_calibration_cut': full,
        'tolerance': TOLERANCE,
        'seeds_required': SEEDS_REQUIRED,
        'counts': rows,
        'smallest_count_within_tolerance': min(enough) if enough else None,
    }


def transfer_report(
    vectors: Any, keys: Any, truth: Any, cut: float, scored: Mask, strata: Any, labelled: Ints
) -> dict[str, Any]:
    """The transferred cut scored on a project's held-out half, beside the trivial baselines.

    ``labelled`` are the rows of ``vectors`` that ``truth``, ``scored`` and ``strata`` describe.
    """
    part = cp.WithinKeyClusterer(vectors, keys).partition(cut)[labelled]
    all_views = cp.views(truth, part, scored, strata)
    return {
        'cut': cut,
        'views': all_views,
        'bar': scorecard.acceptance(
            'bcubed_precision_all',
            TRANSFER_BCUBED_PRECISION_BAR,
            all_views['all']['bcubed_precision'],
        ),
        'detector_class_only': cp.score_block(
            truth, cp.key_only_partition(keys)[labelled], scored, with_ami=False
        ),
    }
