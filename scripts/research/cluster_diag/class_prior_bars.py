"""The held-out protocol for the class-partitioned arm, and the product bars it is held to.

Per fold a cut is chosen on one half of the source images at a target pair recall and scored
on the other half. Each chosen cut is reported on all crops, the non-person crops and every
stratum, beside the refit IVF production configuration and the trivial baselines, so the
figures cannot be read as the person class or the detector alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench import scorecard
from scripts.bench.parity_metrics import adjusted_rand_index
from scripts.research.cluster_diag import class_prior as cp, heldout as ho


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    Ints = NDArray[np.integer[Any]]
    Mask = NDArray[np.bool_]
    Candidates = Sequence[tuple[float, Ints]]
    Views = dict[str, dict[str, float | int | None]]

TARGET_RECALLS = (0.05, 0.1, 0.2, 0.3)
PRODUCT_TARGET = 0.1  # the operating point the product bars are stated at
BCUBED_PRECISION_BAR = 0.90
BCUBED_PRECISION_NON_PERSON_BAR = 0.88
BCUBED_PRECISION_LOW_RES_BAR = 0.85
BCUBED_RECALL_BAR = 0.37  # twice the refit IVF configuration's 0.187, as stated
PAIR_RECALL_BAR = 3 * 0.022
LOW_RES = 'stratum:low_res'


def _value(views: Views, view: str, field: str) -> float | None:
    block = views.get(view)
    return None if block is None else block[field]


def stratum_precision_gaps(within: Views, ivf: Views) -> dict[str, float]:
    """BCubed precision of the arm minus IVF's, per stratum both views scored."""
    return {
        name: round(float(block['bcubed_precision']) - float(ivf[name]['bcubed_precision']), 9)  # type: ignore[arg-type]
        for name, block in within.items()
        if name.startswith('stratum:') and name in ivf
    }


def product_bars(
    within: Views, ivf: Views, *, permutation_ari: float, protected_moved: int
) -> list[dict[str, Any]]:
    """The product bars for one chosen cut, each as a ``scorecard.acceptance`` row."""
    gaps = stratum_precision_gaps(within, ivf)
    worst = min(gaps.items(), key=lambda kv: kv[1]) if gaps else ('none', 0.0)
    bars = [
        scorecard.acceptance(
            'bcubed_precision_all',
            BCUBED_PRECISION_BAR,
            _value(within, 'all', 'bcubed_precision'),
        ),
        scorecard.acceptance(
            'bcubed_precision_non_person',
            BCUBED_PRECISION_NON_PERSON_BAR,
            _value(within, 'non_person', 'bcubed_precision'),
        ),
        scorecard.acceptance(
            'bcubed_precision_low_res',
            BCUBED_PRECISION_LOW_RES_BAR,
            _value(within, LOW_RES, 'bcubed_precision'),
        ),
        scorecard.acceptance(
            'bcubed_recall_all', BCUBED_RECALL_BAR, _value(within, 'all', 'bcubed_recall')
        ),
        scorecard.acceptance(
            'pair_recall_all', PAIR_RECALL_BAR, _value(within, 'all', 'pair_recall')
        ),
        scorecard.acceptance('min_stratum_bcubed_precision_minus_ivf', 0.0, worst[1]),
        scorecard.acceptance('permutation_ari', 1.0, permutation_ari),
        scorecard.acceptance('protected_items_moved', 0, protected_moved, higher_is_better=False),
    ]
    bars[5]['worst_stratum'] = worst[0]
    return bars


def permutation_ari(fit: Any, vectors: Any, keys: Any, ids: Sequence[str], seed: int) -> float:
    """ARI between a fit on the given order and on a seeded permutation of the rows.

    Both fits are sorted by crop id first, which is how the residual pool is fetched.
    """
    n = len(vectors)
    order = np.random.default_rng(seed).permutation(n)
    ids_arr = np.asarray(ids)

    def by_id(rows: Ints) -> Ints:
        return np.asarray(rows[np.argsort(ids_arr[rows], kind='stable')])

    base_rows, perm_rows = by_id(np.arange(n)), by_id(order)
    base = np.empty(n, dtype=np.int64)
    perm = np.empty(n, dtype=np.int64)
    base[base_rows] = fit(vectors[base_rows], np.asarray(keys)[base_rows])
    perm[perm_rows] = fit(vectors[perm_rows], np.asarray(keys)[perm_rows])
    return float(adjusted_rand_index(base, perm))


def _family_rows(
    families: Mapping[str, Candidates],
    truth: Any,
    fit: Mask,
    scored: Mask,
    strata: Mapping[str, Mask],
    targets: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """For each family and target, the cut chosen on ``fit`` and its views on ``scored``."""
    rows: dict[str, dict[str, Any]] = {}
    for family, cands in families.items():
        lookup = dict(cands)
        rows[family] = {}
        for target in targets:
            chosen = ho.choose_and_score(cands, truth, fit, scored, target)
            if chosen is None:
                rows[family][str(target)] = None
                continue
            part = lookup[chosen['parameter']]
            rows[family][str(target)] = {
                'cut': chosen['parameter'],
                'fit_half': chosen['calibration'],
                'views': cp.views(truth, part, scored, strata),
            }
    return rows


def fold_report(
    families: Mapping[str, Candidates],
    references: Mapping[str, Ints],
    truth: Any,
    strata: Mapping[str, Mask],
    first_half: Mask,
    *,
    bar_family: str,
    permutation_ari_value: float,
    protected_moved: int,
    targets: Sequence[float] = TARGET_RECALLS,
) -> dict[str, Any]:
    """Both folds: cuts chosen on one half, every view and the product bars on the other.

    ``families`` are the swept methods (candidate ``(cut, partition)`` lists aligned to
    ``truth``); ``references`` are fixed partitions scored on the same half (IVF, the
    trivial baselines). The bars are attached to ``bar_family`` at every target.
    """
    out: dict[str, Any] = {}
    for name, fit, scored in (
        ('fold1_calibration_to_heldout', first_half, ~first_half),
        ('fold2_heldout_to_calibration', ~first_half, first_half),
    ):
        rows = _family_rows(families, truth, fit, scored, strata, targets)
        ref_views = {k: cp.views(truth, v, scored, strata) for k, v in references.items()}
        for cell in rows[bar_family].values():
            if cell is not None:
                cell['bars'] = product_bars(
                    cell['views'],
                    ref_views['ivf_refit'],
                    permutation_ari=permutation_ari_value,
                    protected_moved=protected_moved,
                )
        out[name] = {
            'fit_labelled': int(fit.sum()),
            'scored_labelled': int(scored.sum()),
            'person_share_of_same_label_pairs': cp.person_pair_share(truth, scored),
            'references': ref_views,
            'families': rows,
        }
    return out


def verdict(bars: Sequence[Mapping[str, Any]]) -> str:
    """PASS when every bar passes, FAIL otherwise."""
    return 'PASS' if all(b['pass'] for b in bars) else 'FAIL'


REOPEN_MARGIN = 0.03


def bar_shortfalls(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """How far each bar is missed (positive) or cleared (negative), in the bar's own units."""
    out = {}
    for row in rows:
        value = row['value']
        lower_is_better = row['metric'] == 'protected_items_moved'
        if value is None:
            out[row['metric']] = float('inf')
        else:
            out[row['metric']] = float(
                value - row['bar'] if lower_is_better else row['bar'] - value
            )
    return out


def bars_missed_by_more_than(
    rows: Sequence[Mapping[str, Any]], margin: float = REOPEN_MARGIN
) -> list[str]:
    """Bars missed by more than ``margin``: the ones that would reopen the decision."""
    return sorted(m for m, short in bar_shortfalls(rows).items() if short > margin)
