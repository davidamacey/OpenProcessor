"""The corrected acceptance bars A1 to A7 and the operating-point rule (decision note section 6).

The operating point is chosen on the calibration half only: the largest target pair recall in
``TARGETS`` whose calibration-half BCubed precision is at least 0.90 both overall and on
non-person crops. The bars are then read once on the held-out half, against the incumbent
(IVF, K = min(512, n // 6), refit on the same vectors) measured on the same half. Nothing here is
tuned after a result: the constants below are the note's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench import cluster_eval_metrics as met, scorecard
from scripts.research.cluster_diag import class_prior as cp, heldout as ho


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    Ints = NDArray[np.integer[Any]]
    Mask = NDArray[np.bool_]
    Views = Mapping[str, Mapping[str, Any]]

TARGETS = (0.05, 0.1, 0.2, 0.3)
PRECISION_BAR = 0.90  # A1, A2 and the operating-point rule
LOW_RES_BAR = 0.85  # A3
RECALL_FACTOR = 2.0  # A4
PAIR_RECALL_FACTOR = 3.0  # A5
STRATUM_MIN_CROPS = 100  # A6 applies to strata at least this large in the scored half
STRATUM_TOLERANCE = 0.02  # A6
ARI_BAR = 1.0  # A7
MIN_EVALUABLE = 30  # a bar on a view with fewer labelled crops is not evaluable (A3 only)
LOW_RES = 'stratum:low_res'


def operating_point(
    candidates: Sequence[tuple[float, Ints]],
    truth: Any,
    fit: Mask,
    targets: Sequence[float] = TARGETS,
    bar: float = PRECISION_BAR,
) -> dict[str, Any] | None:
    """The largest target whose chosen parameter keeps calibration BCubed precision >= ``bar``.

    "Chosen" is ``heldout.choose_and_score``: the most pair-precise candidate on the calibration
    half that reaches the target pair recall there. Returns ``None`` when no target qualifies,
    which fails the method.
    """
    lookup = dict(candidates)
    t = np.asarray(truth)
    nonperson = fit & (t != cp.PERSON)
    for target in sorted(targets, reverse=True):
        chosen = ho.choose_and_score(candidates, truth, fit, ~fit, target)
        if chosen is None:
            continue
        part = lookup[chosen['parameter']]
        overall = met.bcubed_scores(t[fit], part[fit])['bcubed_precision']
        non_person = met.bcubed_scores(t[nonperson], part[nonperson])['bcubed_precision']
        if overall >= bar and non_person >= bar:
            return {
                'target': target,
                'parameter': chosen['parameter'],
                'calibration_bcubed_precision': overall,
                'calibration_bcubed_precision_non_person': non_person,
            }
    return None


def _value(views: Views, view: str, field: str) -> float | None:
    block = views.get(view)
    return None if block is None else block[field]


def stratum_rows(within: Views, incumbent: Views) -> list[dict[str, Any]]:
    """A6 per stratum with at least ``STRATUM_MIN_CROPS`` labelled crops in the scored half."""
    rows = []
    for name, block in within.items():
        if not name.startswith('stratum:') or name not in incumbent:
            continue
        if block['n'] < STRATUM_MIN_CROPS:
            continue
        gap = round(
            float(block['bcubed_precision']) - float(incumbent[name]['bcubed_precision']), 9
        )
        rows.append({'stratum': name, 'n': block['n'], 'gap_to_incumbent': gap})
    return rows


def acceptance_bars(
    within: Views, incumbent: Views, *, permutation_ari: float
) -> list[dict[str, Any]]:
    """A1 to A7 as ``scorecard.acceptance`` rows with an id and an ``evaluable`` flag."""
    inc_recall = _value(incumbent, 'all', 'bcubed_recall')
    inc_pair = _value(incumbent, 'all', 'pair_recall')
    strata = stratum_rows(within, incumbent)
    worst = min(strata, key=lambda r: r['gap_to_incumbent']) if strata else None
    rows = [
        ('A1', 'bcubed_precision_all', PRECISION_BAR, _value(within, 'all', 'bcubed_precision')),
        (
            'A2',
            'bcubed_precision_non_person',
            PRECISION_BAR,
            _value(within, 'non_person', 'bcubed_precision'),
        ),
        (
            'A3',
            'bcubed_precision_low_res',
            LOW_RES_BAR,
            _value(within, LOW_RES, 'bcubed_precision'),
        ),
        (
            'A4',
            'bcubed_recall_vs_incumbent',
            None if inc_recall is None else RECALL_FACTOR * inc_recall,
            _value(within, 'all', 'bcubed_recall'),
        ),
        (
            'A5',
            'pair_recall_vs_incumbent',
            None if inc_pair is None else PAIR_RECALL_FACTOR * inc_pair,
            _value(within, 'all', 'pair_recall'),
        ),
        (
            'A6',
            'worst_large_stratum_precision_minus_incumbent',
            -STRATUM_TOLERANCE,
            0.0 if worst is None else worst['gap_to_incumbent'],
        ),
        ('A7', 'permutation_ari', ARI_BAR - 1e-12, permutation_ari),
    ]
    out = []
    for bar_id, metric, bar, value in rows:
        row = scorecard.acceptance(metric, bar if bar is not None else float('nan'), value)
        row['id'] = bar_id
        row['evaluable'] = bar is not None and value is not None
        out.append(row)
    low = within.get(LOW_RES)
    if low is None or low['n'] < MIN_EVALUABLE:
        out[2]['evaluable'] = False
        out[2]['pass'] = False
    out[5]['worst_stratum'] = None if worst is None else worst['stratum']
    out[5]['strata_checked'] = len(strata)
    return out


def verdict(bars: Sequence[Mapping[str, Any]], operating: Mapping[str, Any] | None) -> str:
    """FAIL when no operating point exists or an evaluable bar fails; INCONCLUSIVE when a bar
    cannot be evaluated and no evaluable bar fails; else PASS."""
    if operating is None:
        return 'FAIL'
    if any(b['evaluable'] and not b['pass'] for b in bars):
        return 'FAIL'
    return 'INCONCLUSIVE' if any(not b['evaluable'] for b in bars) else 'PASS'


def reopen_misses(bars: Sequence[Mapping[str, Any]], margin: float = 0.03) -> list[str]:
    """Bars (by id) missed by more than ``margin`` in the bar's own units (the reopen rule)."""
    out = []
    for b in bars:
        if not b['evaluable'] or b['pass']:
            continue
        if b['bar'] - b['value'] > margin:
            out.append(b['id'])
    return sorted(out)
