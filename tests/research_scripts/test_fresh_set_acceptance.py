"""Y1 and Y2 building blocks: the operating-point rule, the bars A1 to A7 and the label-free cut.

Every expected value is worked out in the comment beside it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from scripts.bench.cluster_eval_metrics import UNASSIGNED
from scripts.research.cluster_diag import acceptance as acc, key_cut_rule as kr
from scripts.research.cluster_diag.y_acceptance import RuleClusterer


def _unit_at(*degrees: float) -> np.ndarray:
    rad = np.radians(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1)


# ---------------------------------------------------------------- label-free cut


def test_kth_neighbour_distance_on_a_line_of_angles() -> None:
    # Six points at 0, 10, ..., 50 degrees; k = 2. Second-nearest angular gap per row:
    # 0 -> 20, 10 -> 10 (0 and 20), 20 -> 10, 30 -> 10, 40 -> 10, 50 -> 20 (40 then 30).
    d = kr.nth_neighbour_distance(_unit_at(0, 10, 20, 30, 40, 50), k=2)
    want = [1 - math.cos(math.radians(a)) for a in (20, 10, 10, 10, 10, 20)]
    assert np.allclose(d, want)
    # median of [20, 10, 10, 10, 10, 20] degrees is 10 degrees
    assert kr.key_scale(_unit_at(0, 10, 20, 30, 40, 50), k=2) == pytest.approx(
        1 - math.cos(math.radians(10))
    )


def test_a_key_without_a_kth_neighbour_has_no_scale() -> None:
    assert kr.key_scale(_unit_at(0, 10, 20), k=3) is None  # 3 rows have only 2 neighbours
    with pytest.raises(ValueError, match='no 3-th neighbour'):
        kr.nth_neighbour_distance(_unit_at(0, 10, 20), k=3)


def test_rule_cuts_scale_with_the_key_and_refuse_small_keys() -> None:
    cuts = kr.rule_cuts({'a': 0.2, 'b': 0.05, 'tiny': None}, 1.5)
    assert cuts == {'a': pytest.approx(0.3), 'b': pytest.approx(0.075), 'tiny': None}


def test_partition_with_cuts_leaves_refused_keys_unassigned_and_codes_unique() -> None:
    keys = np.array(['a', 'a', 'b', 'b', 'c'])

    def cut_key(name: str, height: float) -> np.ndarray:
        return np.array([0, 1]) if name == 'a' else np.array([0, 0])

    out = kr.partition_with_cuts(5, keys, {'a': 0.1, 'b': 0.1, 'c': None}, cut_key)
    # a -> codes 0, 1; b -> offset 2 -> 2, 2; c refused -> UNASSIGNED
    assert out.tolist() == [0, 1, 2, 2, UNASSIGNED]


def test_rule_clusterer_gives_each_key_its_own_height() -> None:
    # Key "tight": 12 points 1 degree apart; key "loose": 12 points 5 degrees apart. The loose key's
    # scale is larger, so one alpha gives it a larger cut, and each key cuts at alpha * its scale.
    tight, loose = _unit_at(*range(12)), _unit_at(*[5 * i for i in range(12)])
    vecs = np.vstack([tight, loose]).astype(np.float32)
    keys = np.array(['tight'] * 12 + ['loose'] * 12)
    rc = RuleClusterer(vecs, keys)
    assert rc.scales['loose'] > rc.scales['tight'] > 0
    cuts = rc.cuts(1.0)
    assert cuts['tight'] == pytest.approx(rc.scales['tight'])
    assert cuts['loose'] == pytest.approx(rc.scales['loose'])
    # A huge alpha puts every key in one cluster; a tiny one leaves every point alone.
    assert len(set(rc.partition(1e6).tolist())) == 2
    assert len(set(rc.partition(1e-9).tolist())) == 24


def test_rule_clusterer_refuses_a_key_of_ten_or_fewer() -> None:
    vecs = np.vstack([_unit_at(*range(12)), _unit_at(0, 1, 2)]).astype(np.float32)
    keys = np.array(['big'] * 12 + ['small'] * 3)
    part = RuleClusterer(vecs, keys).partition(1.0)
    assert (part[12:] == UNASSIGNED).all()
    assert (part[:12] != UNASSIGNED).all()


# ---------------------------------------------------------------- operating point


def _split_classes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.array(['a'] * 10 + ['b'] * 10)
    halves = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)  # each class in two clusters of 5
    all_in_one = np.zeros(20, dtype=np.int64)
    return truth, halves, all_in_one


def test_operating_point_takes_the_largest_target_that_stays_precise() -> None:
    truth, halves, all_in_one = _split_classes()
    # halves: pair recall 2 * 2 * C(5,2) / (2 * C(10,2)) = 40 / 90 = 0.444, BCubed precision 1.
    # all_in_one: pair recall 1, BCubed precision 0.5. At target 0.5 only all_in_one reaches the
    # recall and it is imprecise; at 0.1 the more pair-precise halves wins and qualifies.
    cands = [(1.0, halves), (2.0, all_in_one)]
    fit = np.ones(20, dtype=bool)
    got = acc.operating_point(cands, truth, fit, targets=(0.5, 0.1))
    assert got is not None
    assert got['target'] == 0.1
    assert got['parameter'] == 1.0
    # With a target 0.4 the halves partition reaches recall 0.444 and the larger target qualifies.
    got = acc.operating_point(cands, truth, fit, targets=(0.4, 0.1))
    assert got is not None
    assert got['target'] == 0.4
    assert acc.operating_point([(2.0, all_in_one)], truth, fit, targets=(0.5, 0.1)) is None


def test_operating_point_needs_precision_on_non_person_crops_too() -> None:
    # 100 person crops perfectly clustered, 10 x and 10 y crops in one cluster: overall BCubed
    # precision (100 * 1 + 20 * 0.5) / 120 = 0.917 >= 0.9, non-person 0.5 < 0.9.
    truth = np.array(['person'] * 100 + ['x'] * 10 + ['y'] * 10)
    mixed = np.array([0] * 100 + [1] * 20)
    separated = np.array([0] * 100 + [1] * 10 + [2] * 10)
    fit = np.ones(120, dtype=bool)
    assert acc.operating_point([(1.0, mixed)], truth, fit, targets=(0.1,)) is None
    got = acc.operating_point([(1.0, mixed), (2.0, separated)], truth, fit, targets=(0.1,))
    assert got is not None
    assert got['parameter'] == 2.0


# ---------------------------------------------------------------- bars


def _views(prec: float, recall: float, pair: float, strata: dict[str, tuple[int, float]]) -> Any:
    out: dict[str, dict[str, Any]] = {
        'all': {'bcubed_precision': prec, 'bcubed_recall': recall, 'pair_recall': pair, 'n': 500},
        'non_person': {'bcubed_precision': 0.93, 'n': 300},
    }
    for name, (n, p) in strata.items():
        out[f'stratum:{name}'] = {'bcubed_precision': p, 'n': n}
    return out


def test_recall_bars_are_multiples_of_the_incumbent_measured_on_the_same_half() -> None:
    inc = _views(0.80, 0.20, 0.03, {'low_res': (60, 0.80)})
    within = _views(0.94, 0.41, 0.10, {'low_res': (60, 0.95)})
    rows = {r['id']: r for r in acc.acceptance_bars(within, inc, permutation_ari=1.0)}
    assert rows['A4']['bar'] == pytest.approx(0.40)  # 2 x 0.20
    assert rows['A4']['pass']
    assert rows['A5']['bar'] == pytest.approx(0.09)  # 3 x 0.03
    assert rows['A5']['pass']
    within['all']['bcubed_recall'] = 0.39
    rows = {r['id']: r for r in acc.acceptance_bars(within, inc, permutation_ari=1.0)}
    assert not rows['A4']['pass']


def test_stratum_rule_counts_only_strata_of_100_or_more_crops() -> None:
    inc = _views(0.80, 0.20, 0.03, {'low_res': (60, 0.80), 'big': (150, 0.90), 'tiny': (99, 0.90)})
    within = _views(
        0.94, 0.41, 0.10, {'low_res': (60, 0.95), 'big': (150, 0.89), 'tiny': (99, 0.40)}
    )
    rows = {r['id']: r for r in acc.acceptance_bars(within, inc, permutation_ari=1.0)}
    # big: 0.89 - 0.90 = -0.01, within the 0.02 tolerance; tiny (99 crops, -0.50) is not checked.
    assert rows['A6']['pass']
    assert rows['A6']['strata_checked'] == 1  # only `big` has >= 100 crops
    within['stratum:big']['bcubed_precision'] = 0.87  # -0.03 against the incumbent
    rows = {r['id']: r for r in acc.acceptance_bars(within, inc, permutation_ari=1.0)}
    assert not rows['A6']['pass']
    assert rows['A6']['worst_stratum'] == 'stratum:big'


def test_verdicts_pass_fail_and_inconclusive() -> None:
    inc = _views(0.80, 0.20, 0.03, {'low_res': (60, 0.80)})
    ok = _views(0.94, 0.41, 0.10, {'low_res': (60, 0.95)})
    op = {'target': 0.1, 'parameter': 0.3}
    assert acc.verdict(acc.acceptance_bars(ok, inc, permutation_ari=1.0), op) == 'PASS'
    assert acc.verdict(acc.acceptance_bars(ok, inc, permutation_ari=0.9), op) == 'FAIL'
    assert acc.verdict(acc.acceptance_bars(ok, inc, permutation_ari=1.0), None) == 'FAIL'
    small = _views(0.94, 0.41, 0.10, {'low_res': (10, 0.95)})  # 10 crops: A3 cannot be read
    assert (
        acc.verdict(
            acc.acceptance_bars(
                small, _views(0.8, 0.2, 0.03, {'low_res': (10, 0.8)}), permutation_ari=1.0
            ),
            op,
        )
        == 'INCONCLUSIVE'
    )
    bad = _views(0.80, 0.41, 0.10, {'low_res': (10, 0.95)})
    assert (
        acc.verdict(
            acc.acceptance_bars(
                bad, _views(0.8, 0.2, 0.03, {'low_res': (10, 0.8)}), permutation_ari=1.0
            ),
            op,
        )
        == 'FAIL'
    )


def test_reopen_rule_flags_a_bar_missed_by_more_than_0_03() -> None:
    inc = _views(0.80, 0.20, 0.03, {'low_res': (60, 0.80), 'big': (150, 0.90)})
    within = _views(0.94, 0.41, 0.10, {'low_res': (60, 0.95), 'big': (150, 0.90)})
    within['all']['bcubed_recall'] = 0.38  # bar 0.40: missed by 0.02, not flagged
    assert acc.reopen_misses(acc.acceptance_bars(within, inc, permutation_ari=1.0)) == []
    within['all']['bcubed_recall'] = 0.36  # missed by 0.04
    assert acc.reopen_misses(acc.acceptance_bars(within, inc, permutation_ari=1.0)) == ['A4']


# ---------------------------------------------------------------- one fold, end to end


def _toy_bundle() -> Any:
    """Four tight groups of 12 unit vectors in two detector keys; labels equal the groups."""
    from scripts.research.cluster_diag.dataset import Bundle

    rng = np.random.default_rng(0)
    centres = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    rows = []
    for c in centres:
        v = c + 0.05 * rng.normal(size=(12, 4))
        rows.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    vecs = np.vstack(rows).astype(np.float32)
    names = np.array(['p'] * 24 + ['q'] * 24)
    truth = np.array(['a'] * 12 + ['b'] * 12 + ['c'] * 12 + ['d'] * 12)
    n = len(vecs)
    b = Bundle(
        vectors=vecs,
        meta=[{'crop_id': f'c{i:03d}'} for i in range(n)],
        labelled=np.arange(n),
        truth=truth,
        stratum=np.array(['low_res'] * n),
        group=[f'img{i // 2}' for i in range(n)],
        aspect=np.ones(n),
        side=np.full(n, 20.0),
        protected=np.zeros(n, dtype=bool),
    )
    return b, names


def test_evaluate_fold_reads_the_bars_on_the_other_half_and_honours_a_frozen_parameter() -> None:
    from scripts.research.cluster_diag import y_acceptance as ya

    b, keys = _toy_bundle()
    arm = ya.global_cut_arm(b, keys)
    fit = np.arange(len(keys)) % 4 < 2  # half of every group, so the scored half has all labels
    incumbent = np.arange(len(keys))  # every crop alone: BCubed recall at its floor
    ids = [m['crop_id'] for m in b.meta]
    got = ya.evaluate_fold(b, keys, arm, incumbent, fit, ids=ids, seed=1, frozen=0.30)
    assert got['operating_point']['source'].startswith('frozen')
    assert got['parameter'] == 0.30
    assert got['scored_labelled'] == int((~fit).sum())
    ids_met = {r['id']: r for r in got['bars']}
    assert ids_met['A1']['value'] == 1.0  # four tight groups, each its own cluster at cut 0.3
    assert ids_met['A7']['pass']
    # a cut so low that nothing merges leaves BCubed recall at the incumbent's, below 2x
    low = ya.evaluate_fold(b, keys, arm, incumbent, fit, ids=ids, seed=1, frozen=0.0001)
    assert not {r['id']: r for r in low['bars']}['A4']['pass']
    assert low['verdict'] == 'FAIL'


def test_a_method_with_no_qualifying_target_fails_without_reading_the_scored_half() -> None:
    from scripts.research.cluster_diag import y_acceptance as ya

    b, keys = _toy_bundle()
    only_one_cluster = ya.Arm(
        'blob',
        [(1.0, np.zeros(len(keys), dtype=np.int64))],
        lambda v, _k, _p: np.zeros(len(v), dtype=np.int64),
        lambda _p: np.zeros(len(keys), dtype=np.int64),
    )
    fit = np.arange(len(keys)) % 4 < 2
    got = ya.evaluate_fold(
        b,
        keys,
        only_one_cluster,
        np.arange(len(keys)),
        fit,
        ids=[m['crop_id'] for m in b.meta],
        seed=1,
    )
    assert got['operating_point'] is None
    assert got['verdict'] == 'FAIL'
    assert 'views' not in got
