"""Verdict rules of the registered bars B1 to B4 and the sanity checks (``wb2_standard_benchmarks.md``, sections 4 to 6).

Pure functions over numbers. A bootstrap interval is ``{'delta', 'lo', 'hi'}`` from
``metrics.paired_bootstrap``; a verdict is ``PASS``, ``FAIL`` or ``INCONCLUSIVE``.
"""

from __future__ import annotations

from itertools import combinations, pairwise
from typing import Any


PASS, FAIL, INCONCLUSIVE = 'PASS', 'FAIL', 'INCONCLUSIVE'
MARGIN = 0.03
TOLERANCE = 0.01
MIN_PRECISION = 0.90
PRODUCTION = 'pe_core_l14_336'

# Image encoders with a text tower for the same family of checkpoints (open_clip configurations).
TEXT_TOWER = {
    'pe_core_l14_336': True,
    'pe_core_g14_448': True,
    'mobileclip2_s2': True,
    'siglip2_so400m_p16_256': True,
    'dinov2_l14': False,
}

# Wave WB, Commons set C, held-out half, centre-crop, model-level P@10 (raw JSON of that wave).
SETC_P10 = {
    'pe_core_g14_448': 0.7950,
    'mobileclip2_s2': 0.7590,
    'siglip2_so400m_p16_256': 0.7282,
    'pe_core_l14_336': 0.6498,
    'dinov2_l14': 0.2816,
}


def gain_verdict(cars: dict[str, float], others: dict[str, dict[str, float]]) -> str:
    """B1 and B2: a gain of at least ``MARGIN`` on CARS196 with no loss beyond ``TOLERANCE`` elsewhere.

    ``cars`` is the bootstrap interval of the Recall@1 difference on CARS196, ``others`` the same
    for the other sets.
    """
    if cars['hi'] < MARGIN or any(o['hi'] < -TOLERANCE for o in others.values()):
        return FAIL
    if (
        cars['delta'] >= MARGIN
        and cars['lo'] > 0
        and all(o['delta'] >= -TOLERANCE for o in others.values())
    ):
        return PASS
    return INCONCLUSIVE


def cut_verdict(calibration_point: dict[str, Any] | None, test_precision: float) -> str:
    """B3: a cut qualified on calibration and the test BCubed precision at it reaches ``MIN_PRECISION``."""
    if calibration_point is None or test_precision < MIN_PRECISION:
        return FAIL
    return PASS


def sanity(per_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Section 4 items 2 to 4, from per-model summaries.

    Each model entry holds ``recall`` (``{K: value}`` on CARS196 test), ``control_recall1``,
    ``control_nmi``, ``nmi``, ``rows``, ``expected_rows`` and ``all_recall_max`` (the highest
    Recall@1 over the sets).
    """
    out: dict[str, Any] = {}
    for name, m in per_model.items():
        r = [m['recall'][k] for k in sorted(m['recall'], key=int)]
        out[name] = {
            'control_is_chance': m['control_recall1'] < 0.02 and m['control_nmi'] < 0.05,
            'recall_monotone': all(a <= b for a, b in pairwise(r)),
            'nmi_in_range': 0 < m['nmi'] < 1,
            'rows_match': m['rows'] == m['expected_rows'],
            'recall1_plausible': m['recall']['1'] >= 0.20 and m['all_recall_max'] <= 0.95,
        }
    out['all_pass'] = all(all(v.values()) for v in out.values())
    return out


def weak_ordering(recall1: dict[str, float]) -> dict[str, Any]:
    """Section 4 item 5: SigLIP 2 is at least DINOv2-L on CARS196 (a probe-based expectation, not a gate)."""
    s, d = recall1['siglip2_so400m_p16_256'], recall1['dinov2_l14']
    return {'siglip2': s, 'dinov2': d, 'consistent': s >= d}


def ordering_contradictions(
    recall1: dict[str, float], interval: dict[tuple[str, str], dict[str, float]]
) -> list[dict[str, Any]]:
    """Model pairs whose CARS196 order is the reverse of the set C order with an interval that excludes 0.

    ``interval[(a, b)]`` is the bootstrap interval of Recall@1(a) - Recall@1(b).
    """
    found = []
    for a, b in combinations(sorted(SETC_P10), 2):
        high, low = (
            (a, b) if SETC_P10[a] > SETC_P10[b] else (b, a)
        )  # set C ranks ``high`` above ``low``
        if (high, low) in interval:
            iv = interval[(high, low)]
        elif (low, high) in interval:
            rev = interval[(low, high)]
            iv = {'delta': -rev['delta'], 'lo': -rev['hi'], 'hi': -rev['lo']}
        else:
            continue
        if iv['hi'] < 0:
            found.append(
                {
                    'set_c_higher': high,
                    'cars_higher': low,
                    **iv,
                    'recall1': {high: recall1[high], low: recall1[low]},
                }
            )
    return found
