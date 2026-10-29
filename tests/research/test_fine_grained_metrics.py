"""Fine-grained metrics, the operating-point rule, the ECG weights and the registered bars."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.bench.cluster_eval_metrics import UNASSIGNED
from scripts.research.fine_grained import bars, methods, metrics


def test_ranking_scores_on_a_hand_computed_case() -> None:
    # rows 0,1,2 are class a, row 3 is class b; similarity ranks 0 -> 1, 3, 2
    sim = np.array(
        [
            [1.0, 0.9, 0.5, 0.7],
            [0.9, 1.0, 0.8, 0.1],
            [0.5, 0.8, 1.0, 0.2],
            [0.7, 0.1, 0.2, 1.0],
        ]
    )
    r = metrics.ranking_scores(sim, ['a', 'a', 'a', 'b'], k=2)
    assert r['queries'] == 3  # the lone b row has no same-class peer and is skipped
    assert r['P@1'] == pytest.approx(1.0)  # 0 -> 1 (a), 1 -> 0 (a), 2 -> 1 (a)
    assert r['P@2'] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    # AP@R: row 0 retrieves a, b, a with R=2 -> hits at ranks 1 only -> (1/1)/2 = 0.5
    assert r['mAP@R'] == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_map_at_r_rewards_early_hits() -> None:
    # query 0 ranks x, d, d: R = 2, the top two are [miss, hit], so AP@R = (0 + 1/2) / 2 = 0.25
    # a hit-rate shortcut (mean of the top-R hits) would give 0.5 for it
    sim = np.array(
        [
            [1.0, 0.5, 0.4, 0.9],
            [0.5, 1.0, 0.45, 0.1],
            [0.4, 0.45, 1.0, 0.2],
            [0.9, 0.1, 0.2, 1.0],
        ]
    )
    r = metrics.ranking_scores(sim, ['d', 'd', 'd', 'x'], k=3)
    assert r['mAP@R'] == pytest.approx((0.25 + 1.0 + 1.0) / 3)


def test_ranking_scores_drop_when_neighbours_are_wrong() -> None:
    sim = np.array([[1, 0.1, 0.9], [0.1, 1, 0.2], [0.9, 0.2, 1.0]])
    r = metrics.ranking_scores(sim, ['a', 'a', 'b'], k=1)
    assert r['P@1'] == 0.0


def test_within_make_false_merge_share() -> None:
    part = np.array([0, 0, 0, 0, 1, 1])
    model = ['m1', 'm1', 'm2', 'm3', 'x', 'y']
    make = ['A', 'A', 'A', 'B', 'C', 'D']
    # cluster 0 wrong-model pairs: (0,2) same make, (0,3) no, (1,2) same make, (1,3) no, (2,3) no
    # cluster 1: (x, y) different makes
    assert metrics.within_make_false_merge_share(part, model, make) == pytest.approx(2 / 6)
    assert metrics.within_make_false_merge_share(np.array([0, 1]), ['a', 'b'], ['A', 'A']) is None


def test_review_cost_counts_sheets_odd_cells_and_singletons() -> None:
    part = np.array([0] * 30 + [1] * 4 + [2] + [UNASSIGNED])
    model = ['a'] * 28 + ['b'] * 2 + ['c'] * 3 + ['d'] + ['e'] + ['f']
    # cluster 0: 30 crops -> 2 sheets + 2 odd; cluster 1: 1 sheet + 1 odd; singleton 1; unassigned 1
    assert metrics.review_cost_r1(part, model) == pytest.approx((2 + 2 + 1 + 1 + 1 + 1) / 36)


def test_partition_scores_perfect_and_merged() -> None:
    model = ['a', 'a', 'b', 'b']
    make = ['M', 'M', 'M', 'M']
    perfect = metrics.partition_scores(np.array([0, 0, 1, 1]), model, make)
    assert perfect['model']['bcubed_precision'] == 1.0
    assert perfect['make']['bcubed_precision'] == 1.0
    merged = metrics.partition_scores(np.array([0, 0, 0, 0]), model, make)
    assert merged['model']['bcubed_precision'] == pytest.approx(0.5)
    assert merged['make']['bcubed_precision'] == 1.0  # the merge stays inside one make
    assert merged['within_make_false_merge_share'] == 1.0


def test_operating_point_takes_the_largest_target_that_stays_precise() -> None:
    pts = [
        {'param': 0.1, 'recall': 0.06, 'precision': 0.97},
        {'param': 0.2, 'recall': 0.12, 'precision': 0.93},
        {'param': 0.3, 'recall': 0.22, 'precision': 0.91},
        {'param': 0.4, 'recall': 0.35, 'precision': 0.80},
    ]
    chosen = metrics.choose_operating_point(pts)
    assert chosen is not None
    assert chosen['param'] == 0.3
    assert chosen['target'] == 0.2
    assert metrics.choose_operating_point([{'param': 1, 'recall': 0.5, 'precision': 0.5}]) is None
    assert metrics.choose_operating_point([{'param': 1, 'recall': 0.01, 'precision': 1.0}]) is None


def test_matched_point_respects_the_recall_window() -> None:
    curve = [
        {'recall': 0.30, 'precision': 0.8},
        {'recall': 0.31, 'precision': 0.9},
        {'recall': 0.40, 'precision': 0.99},
    ]
    assert metrics.matched_point(curve, 0.30, 0.02)['precision'] == 0.9
    assert metrics.matched_point(curve, 0.60, 0.02) is None


def test_ecg_edge_weights_are_min_weight_plus_co_association() -> None:
    runs = [[True, False], [True, False], [True, True], [True, False]]
    w = methods.ecg_edge_weights(runs, min_weight=0.05)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.05 + 0.95 * 0.25)


def test_ahc_cuts_split_two_tight_groups() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal([1, 0, 0], 0.01, (6, 3))
    b = rng.normal([0, 1, 0], 0.01, (6, 3))
    v = np.vstack([a, b])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    _, parts = methods.ahc_cuts(methods.cos_sim(v), [0.5])
    assert len(set(parts[0][:6])) == 1
    assert len(set(parts[0][6:])) == 1
    assert parts[0][0] != parts[0][6]


def test_bars_follow_the_registered_thresholds() -> None:
    assert bars.fg1_challenger(0.53, 0.50) == bars.PASS
    assert bars.fg1_challenger(0.529, 0.50) == bars.FAIL
    assert bars.fg1_variant(0.60, 0.55, 0.9, 0.9, None, None) == bars.INCONCLUSIVE
    assert bars.fg1_variant(0.60, 0.55, 0.88, 0.9, 0.7, 0.7) == bars.FAIL  # make P@10 lost 0.02
    assert bars.fg1_variant(0.60, 0.55, 0.9, 0.9, 0.7, 0.7) == bars.PASS
    assert bars.fg2(0.91, 0.96, 0.4, 1.0)['verdict'] == bars.PASS
    assert bars.fg2(0.91, 0.96, 0.6, 1.0)['verdict'] == bars.FAIL
    assert bars.fg2(0.91, 0.96, 0.4, 0.99)['verdict'] == bars.FAIL
    assert bars.fg3(0.96, 0.5, True) == bars.PASS
    assert bars.fg3(0.96, 0.6, True) == bars.FAIL
    assert bars.fg3(0.99, 0.1, False) == bars.INCONCLUSIVE
    good = {'gain': 0.025, 'make_delta': 0.0}
    assert bars.fg4([good, good], None, False)['verdict'] == bars.PASS
    assert bars.fg4([good, {'gain': 0.01, 'make_delta': 0.0}], None, False)['verdict'] == bars.FAIL
    assert bars.fg4([good, {}], None, False)['verdict'] == bars.FAIL
    assert bars.fg4([good, good], 0.9, True)['verdict'] == bars.FAIL
    assert bars.fg5(True, 0.45, 0.5, True) == bars.PASS
    assert bars.fg5(True, 0.46, 0.5, True) == bars.FAIL
    assert bars.fg6({'A': 0.95, 'B': 0.89}, 0.9) == bars.FAIL
    assert bars.fg7(0.7, 0.8) == bars.PASS
    assert bars.fg7(0.69, 0.9) == bars.FAIL
