"""Partition families inside the car key and their scoring on a fold (calibrate, then read once)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster

from scripts.bench.cluster_eval_metrics import assignment_stability
from scripts.research.cluster_diag.diagnostics import LEIDEN_FRACTIONS, linkage_of
from scripts.research.cluster_diag.gate_sim import seeded_runs
from scripts.research.fine_grained import bars, methods
from scripts.research.fine_grained.metrics import (
    choose_operating_point,
    matched_point,
    partition_scores,
    review_cost_r1,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from scripts.research.fine_grained.data import Universe

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]

N_CUTS = 60
GATE_SEEDS = tuple(range(20))
STABILITY_SEEDS = (0, 1, 2, 3)


def ahc_family(sim: Floats, n_cuts: int = N_CUTS) -> tuple[list[float], list[Ints]]:
    """Average-linkage partitions of every row at distance cuts spread over the dendrogram heights."""
    z = linkage_of(sim, 'average')
    heights = [float(h) for h in np.quantile(z[:, 2], np.linspace(0.5, 0.995, n_cuts))]
    return heights, [np.asarray(fcluster(z, h, 'distance') - 1) for h in heights]


def leiden_family(sim: Floats, seed: int = 0) -> tuple[list[float], list[Ints]]:
    return list(LEIDEN_FRACTIONS), [
        methods.leiden_cpm_partition(sim, 30, f, seed=seed) for f in LEIDEN_FRACTIONS
    ]


def ecg_family(sim: Floats, seed: int = 0) -> tuple[list[float], list[Ints]]:
    return [0.0], [methods.ecg_partition(sim, 30, seed=seed)]


def point(part: Ints, u: Universe, rows: np.ndarray) -> dict[str, Any]:
    """Scores of a partition of all rows read on ``rows`` (a pool of labelled crops)."""
    s = partition_scores(part[rows], u.model[rows], u.make[rows])
    return {
        'recall': s['model_pair_recall'] if s['model_pair_recall'] is not None else 0.0,
        'precision': s['model']['bcubed_precision'],
        'bc_recall': s['model']['bcubed_recall'],
        'make_precision': s['make']['bcubed_precision'],
        'scores': s,
    }


def curve(
    parts: list[Ints], params: list[float], u: Universe, rows: np.ndarray
) -> list[dict[str, Any]]:
    return [{'param': p, **point(part, u, rows)} for p, part in zip(params, parts, strict=True)]


def gate(part: Ints, u: Universe, rows: np.ndarray) -> dict[str, Any]:
    """U1 to U3 of the loop's gate at model level and, as a second row, at make level."""
    return {
        'model': seeded_runs(part[rows], u.model[rows].tolist(), seeds=GATE_SEEDS),
        'make': seeded_runs(part[rows], u.make[rows].tolist(), seeds=GATE_SEEDS),
    }


def read_fold(parts: list[Ints], params: list[float], u: Universe, fold: int) -> dict[str, Any]:
    """Choose the operating point on the fold's calibration half, read it on the test half."""
    cal, test = u.fold(fold)
    cal_curve = curve(parts, params, u, cal)
    chosen = choose_operating_point(cal_curve)
    test_curve = curve(parts, params, u, test)
    out: dict[str, Any] = {
        'chosen': None
        if chosen is None
        else {k: chosen[k] for k in ('param', 'target', 'recall', 'precision')}
    }
    out['test_curve'] = [
        {k: c[k] for k in ('param', 'recall', 'precision', 'bc_recall', 'make_precision')}
        for c in test_curve
    ]
    if chosen is None:
        return out
    idx = params.index(chosen['param'])
    part = parts[idx]
    read = test_curve[idx]
    out['test'] = {
        **read['scores'],
        'r1': review_cost_r1(part[test], u.model[test]),
        'gate': gate(part, u, test),
        'precision_cal': chosen['precision'],
    }
    out['part_index'] = idx
    return out


def matched_vs_ahc(read: dict[str, Any], ahc_curve: list[dict[str, Any]]) -> dict[str, float]:
    """FG-4 input: model precision gain and make precision change against average linkage at matched recall."""
    if 'test' not in read:
        return {}
    t = read['test']
    bc_r, bc_p, mk = (
        t['model']['bcubed_recall'],
        t['model']['bcubed_precision'],
        t['make']['bcubed_precision'],
    )
    ref = matched_point(
        [
            {'recall': c['bc_recall'], 'precision': c['precision'], 'make': c['make_precision']}
            for c in ahc_curve
        ],
        bc_r,
        bars.FG4_RECALL_TOL,
    )
    if ref is None:
        return {}
    return {'gain': bc_p - ref['precision'], 'make_delta': mk - ref['make'], 'matched_recall': bc_r}


def permutation_ari(
    build: Callable[[Floats], Ints], vec: Floats, part: Ints, seed: int = 7
) -> float | None:
    """ARI between ``part`` and the partition of the same vectors in a shuffled row order."""
    perm = np.random.default_rng(seed).permutation(len(vec))
    shuffled = build(vec[perm])
    back = np.empty_like(shuffled)
    back[perm] = shuffled
    ari: float | None = assignment_stability(np.asarray(part), back)['ari']
    return ari
