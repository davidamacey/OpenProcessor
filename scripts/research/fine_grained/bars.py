"""The pre-registered bars FG-0 to FG-7 (docs/research/fine_grained_benchmark_plan.md, section 5).

Each function maps measured numbers to PASS, FAIL or INCONCLUSIVE. The thresholds are copied from
the plan as registered on 2026-10-29 and must not be edited after a read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping

PASS, FAIL, INCONCLUSIVE = 'PASS', 'FAIL', 'INCONCLUSIVE'

FG1_GAIN = 0.03
FG1_MAKE_LOSS = 0.01
FG2_MODEL_PRECISION = 0.90
FG2_MAKE_PRECISION = 0.95
FG2_R1 = 0.5
FG3_U2, FG3_U1 = 0.95, 0.5
FG4_GAIN, FG4_MAKE_LOSS, FG4_ARI, FG4_RECALL_TOL = 0.02, 0.01, 0.95, 0.02
FG5_R1_FACTOR = 0.9
FG6_AUROC, FG6_PRECISION = 0.90, 0.80
FG7_KAPPA, FG7_PRECISION = 0.7, 0.8
REOPEN_MARGIN = 0.03


def fg1_challenger(p10_model: float, p10_pe: float) -> str:
    """A challenger representation goes forward only if model-level P@10 >= PE-Core's + 0.03."""
    return PASS if p10_model >= p10_pe + FG1_GAIN - 1e-12 else FAIL


def fg1_variant(
    p10_model: float,
    p10_base: float,
    p10_make: float,
    p10_make_base: float,
    class_p10: float | None,
    class_p10_base: float | None,
) -> str:
    """A preprocessing variant: model P@10 up >= 0.03, make P@10 down <= 0.01, class P@10 down <= 0.01.

    The class-level P@10 on the 3,000-item set is required by the plan; without it the result is
    INCONCLUSIVE rather than PASS.
    """
    if p10_model < p10_base + FG1_GAIN - 1e-12 or p10_make < p10_make_base - FG1_MAKE_LOSS - 1e-12:
        return FAIL
    if class_p10 is None or class_p10_base is None:
        return INCONCLUSIVE
    return PASS if class_p10 >= class_p10_base - FG1_MAKE_LOSS - 1e-12 else FAIL


def fg2(
    model_precision: float, make_precision: float, r1: float, ari: float | None
) -> dict[str, Any]:
    checks = {
        'model_bcubed_precision>=0.90': model_precision >= FG2_MODEL_PRECISION,
        'make_bcubed_precision>=0.95': make_precision >= FG2_MAKE_PRECISION,
        'r1<=0.5': r1 <= FG2_R1,
        'permutation_ari==1.000': ari is not None and ari >= 1.0 - 1e-12,
    }
    return {'checks': checks, 'verdict': PASS if all(checks.values()) else FAIL}


def fg3(u2: float | None, u1: float | None, readable: bool) -> str:
    if not readable or u2 is None or u1 is None:
        return INCONCLUSIVE
    return PASS if u2 >= FG3_U2 and u1 <= FG3_U1 else FAIL


def fg4(
    folds: list[Mapping[str, float]], seed_ari: float | None, needs_ari: bool
) -> dict[str, Any]:
    """Replace average linkage only if, in BOTH folds, matched-recall precision gain >= 0.02 and make loss <= 0.01.

    Each fold is ``{'gain': model precision gain at matched recall, 'make_delta': make precision
    change}``; a fold with no matched point is ``{}`` and fails. Leiden and ECG also need ARI >= 0.95.
    """
    ok = all(
        f and f['gain'] >= FG4_GAIN - 1e-12 and f['make_delta'] >= -FG4_MAKE_LOSS - 1e-12
        for f in folds
    )
    if needs_ari:
        ok = ok and seed_ari is not None and seed_ari >= FG4_ARI
    return {'verdict': PASS if ok else FAIL, 'folds': folds, 'seed_ari': seed_ari}


def fg5(
    passes_fg1: bool, r1_second: float | None, r1_pe: float | None, meets_fg2_precision: bool
) -> str:
    """A second vector: must pass FG-1 and give R1 <= 0.9 x PE-Core's at FG-2's precision bars."""
    if not passes_fg1:
        return FAIL
    if r1_second is None or r1_pe is None or not meets_fg2_precision:
        return FAIL
    return PASS if r1_second <= FG5_R1_FACTOR * r1_pe + 1e-12 else FAIL


def fg6(auroc_by_make: Mapping[str, float], flag_precision: float | None) -> str:
    """Novelty queue: AUROC >= 0.90 in each make with held-out models, and flag precision >= 0.80."""
    if not auroc_by_make or flag_precision is None:
        return INCONCLUSIVE
    ok = all(a >= FG6_AUROC for a in auroc_by_make.values()) and flag_precision >= FG6_PRECISION
    return PASS if ok else FAIL


def fg7(kappa: float | None, flag_precision: float | None) -> str:
    if kappa is None or flag_precision is None:
        return INCONCLUSIVE
    return PASS if kappa >= FG7_KAPPA and flag_precision >= FG7_PRECISION else FAIL


def reopen(fold2: Mapping[str, float], bars: Mapping[str, float]) -> list[str]:
    """Bars (higher is better) missed by more than 0.03 in fold 2: the decision is reopened."""
    return [k for k, bar in bars.items() if fold2.get(k, float('nan')) < bar - REOPEN_MARGIN]
