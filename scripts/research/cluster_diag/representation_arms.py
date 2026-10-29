"""Representation arms inside the detector class: does re-representing the vectors help?

Each arm replaces the cosine distance used by the per-class average-linkage dendrograms:

* ``raw`` - 1 minus the cosine of the stored vectors;
* ``pca_whiten_256`` - PCA whitening to 256 axes, fitted on an independent pool of vectors;
* ``kreciprocal`` - k-reciprocal re-ranking (k1 20, k2 6, lambda 0.3) computed inside the class;
* ``dba`` - database-side augmentation (k 5, alpha 3) computed inside the class.

The re-ranking arms see only the crops of their own class, which is what a product that
clusters per class would hand them. A class of one or two crops keeps the raw distance.
An arm is compared with ``raw`` at matched BCubed recall: its cut is the one whose calibration
BCubed recall is closest to the raw arm's, and the score is on the other half.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster

from scripts.bench.cluster_eval_metrics import UNASSIGNED
from scripts.research.cluster_diag import (
    class_prior as cp,
    class_prior_bars as bars,
    diagnostics as dg,
    heldout as ho,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from scripts.research.cluster_diag.dataset import Bundle

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]
    Mask = NDArray[np.bool_]

ARMS = ('raw', 'pca_whiten_256', 'kreciprocal', 'dba')
MIN_CLASS_FOR_RERANK = 3
RECALL_MATCH = 0.02
PRECISION_GAIN = 0.02
STRATUM_LOSS = 0.02
GRID_POINTS = 150


def _cosine_distance(z: Floats) -> Floats:
    """1 minus the cosine, floored at 0 (identical rows can round to a tiny negative)."""
    return np.asarray(np.clip(1.0 - z @ z.T, 0.0, None))


def distance_fn(arm: str, pool: Floats | None) -> Callable[[Floats], Floats]:
    """The square distance matrix an arm gives to the vectors of one class."""
    if arm == 'raw':
        return _cosine_distance
    if arm == 'pca_whiten_256':
        if pool is None:
            raise ValueError('pca_whiten_256 needs a fit pool')
        mu, proj = dg.pca_whiten_fit(pool, 256)
        return lambda v: _cosine_distance(dg.pca_whiten_apply(v, mu, proj))
    if arm == 'kreciprocal':
        return lambda v: np.clip(-dg.kreciprocal_similarity(v), 0.0, None)
    if arm == 'dba':
        return lambda v: _cosine_distance(_dba_unit_rows(v))
    raise ValueError(f'unknown arm {arm!r}')


def _dba_unit_rows(v: Floats, k: int = 5, alpha: float = 3.0) -> Floats:
    """Rows replaced by their cos**alpha weighted k-nearest mean (the rows of ``dba_similarity``)."""
    sim = v @ v.T
    idx = np.argsort(-sim, axis=1, kind='stable')[:, :k]
    w = np.take_along_axis(sim, idx, 1).clip(0) ** alpha
    return dg.l2_normalize(np.einsum('nk,nkd->nd', w, v[idx]))


class ArmClusterer:
    """Per-key dendrograms under one arm's distance, cut at any height."""

    def __init__(self, vectors: Floats, keys: Any, arm: str, pool: Floats | None = None) -> None:
        self.n = len(vectors)
        dist_of = distance_fn(arm, pool)
        raw = distance_fn('raw', None)
        k = np.asarray(keys).astype(str)
        self._linkages: list[tuple[Ints, Floats | None]] = []
        for name in sorted(set(k.tolist())):
            idx = np.nonzero(k == name)[0]
            if len(idx) == 1:
                self._linkages.append((idx, None))
                continue
            rerank = len(idx) >= MIN_CLASS_FOR_RERANK
            dist = (dist_of if rerank else raw)(vectors[idx])
            self._linkages.append((idx, dg.linkage_from_distance(dist, 'average')))

    def heights(self) -> Floats:
        return np.concatenate([z[:, 2] for _, z in self._linkages if z is not None])

    def grid(self) -> list[float]:
        """Cut heights at evenly spaced quantiles of every merge height of the arm."""
        return sorted(
            {float(t) for t in np.quantile(self.heights(), np.linspace(0.3, 0.995, GRID_POINTS))}
        )

    def partition(self, cut: float) -> Ints:
        labels = np.full(self.n, UNASSIGNED, dtype=np.int64)
        offset = 0
        for idx, z in self._linkages:
            part = (
                np.zeros(len(idx), dtype=np.int64)
                if z is None
                else fcluster(z, cut, 'distance') - 1
            )
            labels[idx] = part + offset
            offset += int(part.max()) + 1
        return labels


def matched_cut(
    cands: Sequence[tuple[float, Ints]], truth: Any, fit: Mask, target_recall: float
) -> float:
    """The candidate whose BCubed recall on ``fit`` is closest to ``target_recall``.

    Ties go to the more precise one on ``fit``.
    """
    scored = [
        (
            abs(float(b['bcubed_recall'] or 0.0) - target_recall),
            -float(b['bcubed_precision'] or 0.0),
            cut,
        )
        for cut, part in cands
        for b in [cp.score_block(truth, part, fit)]
    ]
    return min(scored)[2]


def compare_at_matched_recall(
    arm_cands: Sequence[tuple[float, Ints]],
    raw_part: Ints,
    truth: Any,
    strata: Mapping[str, Mask],
    fit: Mask,
    scored: Mask,
) -> dict[str, Any]:
    """One arm against raw at raw's operating point: precision gain, recall gap, strata losses."""
    raw_fit = cp.score_block(truth, raw_part, fit)
    cut = matched_cut(arm_cands, truth, fit, float(raw_fit['bcubed_recall']))  # type: ignore[arg-type]
    arm_part = dict(arm_cands)[cut]
    raw_views = cp.views(truth, raw_part, scored, strata)
    arm_views = cp.views(truth, arm_part, scored, strata)
    gain = float(arm_views['all']['bcubed_precision']) - float(raw_views['all']['bcubed_precision'])  # type: ignore[arg-type]
    gap = float(arm_views['all']['bcubed_recall']) - float(raw_views['all']['bcubed_recall'])  # type: ignore[arg-type]
    losses = {
        name: float(raw_views[name]['bcubed_precision']) - float(block['bcubed_precision'])  # type: ignore[arg-type]
        for name, block in arm_views.items()
        if name.startswith('stratum:') and name in raw_views
    }
    worst = max(losses.items(), key=lambda kv: kv[1]) if losses else ('none', 0.0)
    raw_recall = float(raw_views['all']['bcubed_recall'])  # type: ignore[arg-type]
    near = [
        float(block['bcubed_precision'])  # type: ignore[arg-type]
        for _, part in arm_cands
        for block in [cp.score_block(truth, part, scored)]
        if abs(float(block['bcubed_recall']) - raw_recall) <= RECALL_MATCH  # type: ignore[arg-type]
    ]
    raw_precision = float(raw_views['all']['bcubed_precision'])  # type: ignore[arg-type]
    return {
        'best_case_gain': max(near) - raw_precision if near else None,
        'arm_cut': cut,
        'raw': raw_views['all'],
        'arm': arm_views['all'],
        'bcubed_precision_gain': gain,
        'bcubed_recall_gap': gap,
        'matched': abs(gap) <= RECALL_MATCH,
        'worst_stratum': worst[0],
        'worst_stratum_loss': worst[1],
        'passes': abs(gap) <= RECALL_MATCH and gain >= PRECISION_GAIN and worst[1] <= STRATUM_LOSS,
    }


def arm_verdict(cells: Sequence[Mapping[str, Any]]) -> str:
    """PASS if every cell passes; FAIL if a cell misses a bar, or is unmatched and could not gain.

    ``best_case_gain`` is the gain at the held-out recall nearest raw's over every cut of the arm
    (an upper bound, never used to pass an arm): below the bar, no matching could have helped.
    """
    if all(c['passes'] for c in cells):
        return 'PASS'

    def hopeless(c: Mapping[str, Any]) -> bool:
        best = c['best_case_gain']
        return best is not None and best < PRECISION_GAIN

    if any((c['matched'] and not c['passes']) or (not c['matched'] and hopeless(c)) for c in cells):
        return 'FAIL'
    return 'INCONCLUSIVE'


def run_representations(
    b: Bundle,
    keys: Any,
    first_half: Mask,
    pool: Floats,
    targets: Sequence[float] = bars.TARGET_RECALLS,
) -> dict[str, Any]:
    """Every non-raw arm against raw at raw's calibration-chosen cuts, both folds."""
    lab, strata = b.labelled, b.strata_masks()
    raw = ArmClusterer(b.vectors, keys, 'raw')
    raw_cands = [(c, raw.partition(c)[lab]) for c in raw.grid()]
    out: dict[str, Any] = {}
    for arm in ARMS[1:]:
        clusterer = ArmClusterer(b.vectors, keys, arm, pool)
        cands = [(c, clusterer.partition(c)[lab]) for c in clusterer.grid()]
        folds: dict[str, dict[str, Any]] = {}
        for name, fit, scored in (
            ('fold1_calibration_to_heldout', first_half, ~first_half),
            ('fold2_heldout_to_calibration', ~first_half, first_half),
        ):
            cells = {}
            for target in targets:
                chosen = ho.choose_and_score(raw_cands, b.truth, fit, scored, target)
                if chosen is not None:
                    raw_part = dict(raw_cands)[chosen['parameter']]
                    cells[str(target)] = {
                        'raw_cut': chosen['parameter'],
                        **compare_at_matched_recall(cands, raw_part, b.truth, strata, fit, scored),
                    }
            folds[name] = cells
        primary = [f[str(bars.PRODUCT_TARGET)] for f in folds.values()]
        out[arm] = {
            'folds': folds,
            'verdict_at_product_target': arm_verdict(primary),
            'verdict_all_targets': arm_verdict([c for f in folds.values() for c in f.values()]),
        }
    return out
