"""The judge-gated merge simulation on a four-leaf dendrogram worked out by hand.

Leaves 0..3. Linkage rows: node 4 = (0, 1) at 0.1; node 5 = (2, 3) at 0.2; node 6 = (4, 5) at 0.5.
At cut 0.15 the units are {0,1}, {2}, {3}.
"""

from __future__ import annotations

import numpy as np

from scripts.research.cluster_diag.merge_stage import Judge, merge_key, merged_partition


Z = np.array([[0, 1, 0.1, 2], [2, 3, 0.2, 2], [4, 5, 0.5, 4]], dtype=float)
ALWAYS = Judge(p_same=1.0, p_false_same=1.0)
NEVER = Judge(p_same=0.0, p_false_same=0.0)
SOUND = Judge(p_same=1.0, p_false_same=0.0)


def test_a_window_below_the_next_merge_checks_nothing() -> None:
    res = merge_key(Z, ['a', 'a', 'a', 'b'], 0.15, 0.04, ALWAYS, 0)  # reaches 0.19 < 0.2
    assert (res.checks, res.accepted) == (0, 0)
    assert res.codes.tolist() == [0, 0, 1, 2]


def test_a_sound_judge_rejects_a_different_pair() -> None:
    # units {2} = a and {3} = b differ; window 0.1 reaches only node 5 (0.2)
    res = merge_key(Z, ['a', 'a', 'a', 'b'], 0.15, 0.1, SOUND, 0)
    assert (res.checks, res.accepted) == (1, 0)
    assert len(set(res.codes.tolist())) == 3


def test_a_sound_judge_accepts_a_same_pair_and_passes_it_up() -> None:
    # labels a a a a: node 5 is the same class and is accepted; node 6 (0.5, window 0.4 reaches
    # 0.55) then compares {0,1} with {2,3}: the same class again
    res = merge_key(Z, ['a', 'a', 'a', 'a'], 0.15, 0.4, SOUND, 0)
    assert (res.checks, res.accepted, res.true_same_accepted) == (2, 2, 2)
    assert set(res.codes.tolist()) == {res.codes[0]}


def test_a_rejected_node_blocks_its_ancestors() -> None:
    res = merge_key(Z, ['a', 'a', 'a', 'b'], 0.15, 0.4, SOUND, 0)
    # node 5 rejected (a against b) so node 6 cannot be checked: one check only
    assert (res.checks, res.accepted) == (1, 0)


def test_a_judge_that_always_says_same_merges_a_wrong_pair_and_the_count_shows_it() -> None:
    res = merge_key(Z, ['a', 'a', 'a', 'b'], 0.15, 0.4, ALWAYS, 0)
    # node 5: a against b, wrong merge. Node 6: {0,1} = a against {2,3} = {a, b} whose plurality
    # is a (tie to the smaller name), so it is a same pair
    assert (res.checks, res.accepted) == (2, 2)
    assert (res.true_same_accepted, res.false_same_accepted) == (1, 1)


def test_a_unit_with_no_labelled_member_cannot_be_checked() -> None:
    res = merge_key(Z, ['a', 'a', None, 'b'], 0.15, 0.4, ALWAYS, 0)
    assert (res.checks, res.accepted) == (0, 0)


def test_merged_partition_keeps_codes_unique_across_keys() -> None:
    linkages = [(np.array([0, 1, 2, 3]), Z), (np.array([4]), None), (np.array([5, 6]), Z[:1])]
    labels = ['a', 'a', 'a', 'a', 'b', 'c', 'c']
    codes, totals = merged_partition(linkages, 7, labels, 0.15, 0.4, SOUND, 0)
    # key 0 merges fully (all a); key 1 is a single row; key 2 is {5,6} joined at 0.1 (<= cut)
    assert len(set(codes[:4].tolist())) == 1
    assert len(set(codes.tolist())) == 3
    assert totals['accepted'] == 2


def test_judge_rates_are_applied_per_truth() -> None:
    rng = np.random.default_rng(0)
    j = Judge(p_same=0.8, p_false_same=0.1)
    same = np.mean([j.says_same(True, rng) for _ in range(4000)])
    diff = np.mean([j.says_same(False, rng) for _ in range(4000)])
    assert abs(same - 0.8) < 0.03
    assert abs(diff - 0.1) < 0.03
