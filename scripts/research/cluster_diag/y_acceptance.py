"""Arms and the fold evaluation behind Y2 and Y1: a swept parameter, an operating point chosen on
the calibration half, and the bars A1 to A7 read on the other half.

Two arms share the code. ``global_cut``: one cosine-distance cut for every key (the design as the
earlier class-partition experiments measured it). ``rule``: each key's cut is alpha times its own
median 10th-neighbour distance (``key_cut_rule``); alpha is the swept parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster

from scripts.research.cluster_diag import (
    acceptance as acc,
    class_prior as cp,
    class_prior_bars as bars,
    key_cut_rule as kr,
    partitions as pt,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from scripts.research.cluster_diag.dataset import Bundle

    Ints = NDArray[np.integer[Any]]
    Floats = NDArray[np.floating[Any]]
    Mask = NDArray[np.bool_]

GLOBAL_CUTS = (0.15, 0.60, 0.005)  # lo, hi, step of the uniform grid of cut heights
ALPHAS = (0.5, 3.0, 0.025)  # lo, hi, step of the grid of rule factors


class RuleClusterer:
    """One dendrogram per key; the cut of each key is ``alpha`` times that key's scale."""

    def __init__(self, vectors: Floats, keys: Any) -> None:
        self.n = len(vectors)
        self.keys = np.asarray(keys).astype(str)
        self.names = sorted(set(self.keys.tolist()))
        self.scales = kr.key_scales(vectors, self.keys)
        self._z = {
            name: z
            for name, (_, z) in zip(
                self.names, pt.group_linkages(self.keys, vectors @ vectors.T), strict=True
            )
        }

    def cuts(self, alpha: float) -> dict[str, float | None]:
        return kr.rule_cuts(self.scales, alpha)

    def partition(self, alpha: float) -> Ints:
        def cut_key(name: str, height: float) -> Ints:
            z = self._z[name]
            n_rows = int((self.keys == name).sum())
            return (
                np.zeros(n_rows, dtype=np.int64)
                if z is None
                else np.asarray(fcluster(z, height, 'distance'), dtype=np.int64) - 1
            )

        return kr.partition_with_cuts(self.n, self.keys, self.cuts(alpha), cut_key)


@dataclass(frozen=True)
class Arm:
    """A swept method: candidate partitions over the labelled crops and how to refit one."""

    name: str
    candidates: list[tuple[float, Ints]]
    refit: Callable[[Floats, Any, float], Ints]  # (vectors, keys, parameter) -> all-crop codes
    full: Callable[[float], Ints]  # parameter -> all-crop codes on the evaluated bundle


def uniform(lo_hi_step: tuple[float, float, float]) -> list[float]:
    return cp.uniform_cut_grid(*lo_hi_step)


def global_cut_arm(b: Bundle, keys: Any) -> Arm:
    clusterer = cp.WithinKeyClusterer(b.vectors, keys)
    return Arm(
        'global_cut',
        [(c, clusterer.partition(c)[b.labelled]) for c in uniform(GLOBAL_CUTS)],
        lambda v, k, c: cp.WithinKeyClusterer(v, k).partition(c),
        clusterer.partition,
    )


def rule_arm(b: Bundle, keys: Any, alphas: Sequence[float] | None = None) -> Arm:
    clusterer = RuleClusterer(b.vectors, keys)
    grid = list(alphas) if alphas is not None else uniform(ALPHAS)
    return Arm(
        'rule',
        [(a, clusterer.partition(a)[b.labelled]) for a in grid],
        lambda v, k, a: RuleClusterer(v, k).partition(a),
        clusterer.partition,
    )


def evaluate_fold(
    b: Bundle,
    keys: Any,
    arm: Arm,
    incumbent: Ints,
    fit: Mask,
    *,
    ids: Sequence[str],
    seed: int,
    frozen: float | None = None,
) -> dict[str, Any]:
    """Operating point on ``fit`` (or the frozen parameter), bars on the other half.

    ``incumbent`` holds the IVF partition of the labelled crops, refit on the same vectors.
    """
    scored = ~fit
    if frozen is None:
        op = acc.operating_point(arm.candidates, b.truth, fit)
        source = 'chosen on the calibration half'
    else:
        op = {'target': None, 'parameter': frozen}
        source = 'frozen from an earlier calibration half'
    if op is None:
        return {
            'arm': arm.name,
            'operating_point': None,
            'verdict': 'FAIL',
            'reason': 'no target recall keeps calibration BCubed precision >= 0.90',
        }
    param = float(op['parameter'])
    part = arm.full(param)[b.labelled]
    strata = b.strata_masks()
    views = cp.views(b.truth, part, scored, strata)
    inc_views = cp.views(b.truth, incumbent, scored, strata)
    ari = bars.permutation_ari(lambda v, k: arm.refit(v, k, param), b.vectors, keys, ids, seed)
    rows = acc.acceptance_bars(views, inc_views, permutation_ari=ari)
    return {
        'arm': arm.name,
        'operating_point': {**op, 'source': source},
        'parameter': param,
        'scored_labelled': int(scored.sum()),
        'views': views,
        'incumbent_views': inc_views,
        'bars': rows,
        'verdict': acc.verdict(rows, op),
        'reopen_misses_over_0.03': acc.reopen_misses(rows),
    }


def frontier(b: Bundle, arm: Arm, fit: Mask, every: int = 4) -> list[dict[str, float]]:
    """BCubed precision and recall of every ``every``-th candidate on both halves (a curve to read,
    not a selection: nothing is chosen from it)."""
    from scripts.bench import cluster_eval_metrics as met

    t = np.asarray(b.truth)
    rows = []
    for param, part in arm.candidates[::every]:
        row = {'parameter': float(param)}
        for name, mask in (('fit', fit), ('other', ~fit)):
            bc = met.bcubed_scores(t[mask], part[mask])
            row[f'{name}_bcubed_precision'] = bc['bcubed_precision']
            row[f'{name}_bcubed_recall'] = bc['bcubed_recall']
            row[f'{name}_pair_recall'] = met.pairwise_scores(t[mask], part[mask])['pair_recall']
        rows.append(row)
    return rows
