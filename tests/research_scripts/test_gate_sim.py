"""The loop-gate simulator (Z1, Z2): every expected value is worked out beside it.

The gate numbers are the auto-label loop's own published tables (theta 0.9, alpha_stage 0.025).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.research.cluster_diag import gate_sim as gs


class InOrder:
    """A generator stub: draws the members in their listed order, answers always right."""

    def permutation(self, n: int) -> np.ndarray:
        return np.arange(n)

    def random(self) -> float:
        return 1.0  # never below a noise probability < 1

    def integers(self, _high: int) -> int:
        return 0


def _cluster(n: int, others: dict[int, str]) -> list[str]:
    return [others.get(i, 'car') for i in range(n)]


def test_required_sample_matches_the_loops_published_table() -> None:
    sizes = (30, 60, 150, 500, 2000, 8000)
    assert [gs.required_sample(n, 0.9, 0.025, 0) for n in sizes] == [18, 24, 30, 34, 35, 35]
    assert [gs.required_sample(n, 0.9, 0.025, 1) for n in sizes] == [24, 34, 44, 51, 53, 54]
    assert gs.required_sample(10**6, 0.9, 0.025, 0) == math.ceil(math.log(0.025) / math.log(0.9))


def test_a_tiny_cluster_is_asked_whole_and_propagates_nothing() -> None:
    # 5 members, theta 0.9: even 5 of 5 agreeing is the only proof, so all 5 are asked
    res = gs.run_gate(_cluster(5, {}), InOrder())
    assert (res.sampled, res.passed, res.propagated) == (5, True, ())


def test_a_pure_cluster_passes_on_the_first_stage_sample() -> None:
    # 150 members -> sample 30 with no dissent; the other 120 take the label
    res = gs.run_gate(_cluster(150, {}), InOrder())
    assert (res.sampled, res.passed, res.top) == (30, True, 'car')
    assert res.propagated == tuple(range(30, 150))


def test_one_dissent_in_the_first_look_triggers_the_top_up() -> None:
    # dog at position 5 is in the first 30. One dissent needs 44 in total, so 14 more are asked
    # (positions 30..43) and all agree: 44 asked, 106 propagated, the dog was asked so no error
    res = gs.run_gate(_cluster(150, {5: 'dog'}), InOrder())
    assert (res.sampled, res.passed, res.top) == (44, True, 'car')
    assert res.propagated == tuple(range(44, 150))


def test_an_unasked_minority_member_is_propagated_wrongly() -> None:
    res = gs.run_gate(_cluster(150, {5: 'dog', 100: 'dog'}), InOrder())
    assert res.passed
    assert 100 in res.propagated  # it takes 'car'; the simulator scores it as an error


def test_two_dissents_in_the_first_look_fail_without_a_top_up() -> None:
    res = gs.run_gate(_cluster(150, {1: 'dog', 2: 'dog'}), InOrder())
    assert (res.passed, res.sampled, res.propagated) == (False, 150, ())


def test_a_dissent_found_only_in_the_top_up_fails() -> None:
    # dog at 5 (first look) and at 35 (top-up positions 30..43): dissent 2 after the second look
    res = gs.run_gate(_cluster(150, {5: 'dog', 35: 'dog'}), InOrder())
    assert (res.passed, res.sampled) == (False, 150)


def test_a_mixed_cluster_fails_and_costs_every_member() -> None:
    truth = ['car'] * 60 + ['dog'] * 90
    res = gs.run_gate(truth, np.random.default_rng(0))
    assert (res.passed, res.sampled, res.propagated) == (False, 150, ())


def test_a_wrong_labeler_makes_the_propagated_label_wrong() -> None:
    # every answer is wrong (noise 1) and the only other class is 'dog': the cluster passes as 'dog'
    res = gs.run_gate(
        _cluster(150, {}), np.random.default_rng(1), noise=1.0, classes=['car', 'dog']
    )
    assert (res.passed, res.top) == (True, 'dog')


def test_simulate_counts_decisions_and_propagated_precision() -> None:
    # cluster 0: 150 cars, passes with 30 asked (120 propagated, all right);
    # cluster 1: dog, cat, cat is below any proof: asked whole (3). Total 33 of 153.
    truth = ['car'] * 150 + ['dog', 'cat', 'cat']
    codes = np.array([0] * 150 + [1] * 3)
    out = gs.simulate(codes, truth, seed=0, open_set={'dog'})
    assert out['decisions'] == 33
    assert out['u1'] == pytest.approx(33 / 153)
    assert out['propagated'] == 120
    assert out['u2'] == 1.0
    assert out['u3'] == pytest.approx(120 / 153)
    assert out['open_propagated'] == 0
    assert math.isnan(out['u2_open'])


def test_simulate_scores_open_set_items_separately() -> None:
    # 200 'goat' items sitting in one 'sheep'-looking cluster of 220 (20 sheep): the gate sees 91 %
    # goat, so a pass is possible; whatever it does, a propagated goat takes the top class, and
    # the open-set count is the propagated goats
    truth = ['goat'] * 200 + ['sheep'] * 20
    out = gs.simulate(np.zeros(220, dtype=int), truth, seed=3, open_set={'goat'})
    if out['propagated']:
        assert out['open_propagated'] <= out['propagated']
        assert out['open_correct'] <= out['open_propagated']


def test_the_result_does_not_depend_on_the_order_clusters_are_listed() -> None:
    truth = ['car'] * 150 + ['dog'] * 150
    a = gs.simulate(np.array([0] * 150 + [1] * 150), truth, seed=7)
    b = gs.simulate(np.array([1] * 150 + [0] * 150), truth[::-1], seed=7)
    assert a['decisions'] == b['decisions']
    assert a['propagated'] == b['propagated']


def test_gate_reach_counts_clusters_that_can_ever_propagate() -> None:
    # 3 and 9 members are asked whole (9: C(8,n)/C(9,n) = (9-n)/9 > 0.025 until n = 9). 10 members
    # prove it at 9 asked, so they can propagate 1, but 2 * 9 > 10 so the sample is over half.
    # 200 and 6000: sample 32 and 35, far under half.
    out = gs.gate_reach([3, 9, 10, 200, 6000])
    assert out['clusters_that_can_propagate'] == 3
    assert out['smallest_cluster_that_can_propagate'] == 10
    assert out['items_in_them'] == 6210
    assert out['clusters_where_the_sample_is_at_most_half'] == 2
    assert out['items'] == 6222
    assert out['largest'] == 6000


def test_seeded_runs_reports_mean_min_max() -> None:
    truth = ['car'] * 150
    out = gs.seeded_runs(np.zeros(150, dtype=int), truth, seeds=[0, 1, 2])
    assert out['u1'] == pytest.approx(
        {'mean': 0.2, 'min': 0.2, 'max': 0.2}
    )  # 30 of 150, every seed
    assert out['u2']['mean'] == 1.0
