"""Cluster-evaluation metrics on synthetic partitions with planted structure."""

from __future__ import annotations

import time

import numpy as np
import pytest

from scripts.bench import cluster_eval_metrics as m


U = m.UNASSIGNED


def _arr(*xs: int) -> np.ndarray:
    return np.array(xs, dtype=np.int64)


def test_perfect_partition_has_no_false_merges_or_splits() -> None:
    truth = _arr(0, 0, 0, 1, 1, 2)
    s = m.pairwise_scores(truth, _arr(7, 7, 7, 3, 3, 5))
    assert s['pair_precision'] == 1.0
    assert s['pair_recall'] == 1.0
    assert s['false_merge_rate'] == 0.0
    assert s['false_split_rate'] == 0.0
    assert s['correct_split_recovery'] == 1.0
    assert s['unassigned_fraction'] == 0.0
    assert s['mixed_clusters'] == 0
    assert s['minority_item_rate'] == 0.0


def test_false_merge_and_false_split_are_reported_separately() -> None:
    truth = _arr(0, 0, 0, 1, 1, 1)
    s = m.pairwise_scores(truth, _arr(0, 0, 1, 0, 2, 2))
    # cluster0={A,A,B}: pairs 3, tp 1, fp 2 ; cluster1={A}; cluster2={B,B}: pairs 1, tp 1
    # predicted pairs 4, tp 2 -> precision 1/2 ; same-class pairs 6 -> recall 1/3
    assert s['pair_precision'] == pytest.approx(2 / 4)
    assert s['false_merge_rate'] == pytest.approx(2 / 4)
    assert s['pair_recall'] == pytest.approx(2 / 6)
    assert s['false_split_rate'] == pytest.approx(4 / 6)
    assert s['mixed_clusters'] == 1
    assert s['minority_item_rate'] == pytest.approx(1 / 6)
    # 9 different-class pairs, 2 of them merged
    assert s['correct_split_recovery'] == pytest.approx(7 / 9)


def test_all_in_one_cluster_is_all_false_merge_no_false_split() -> None:
    truth = _arr(0, 0, 1, 1)
    s = m.pairwise_scores(truth, _arr(0, 0, 0, 0))
    assert s['false_split_rate'] == 0.0
    assert s['false_merge_rate'] == pytest.approx(4 / 6)
    assert s['correct_split_recovery'] == 0.0


def test_unassigned_items_are_missed_pairs_not_merged_pairs() -> None:
    truth = _arr(0, 0, 0, 1, 1)
    s = m.pairwise_scores(truth, _arr(0, 0, U, 1, U))
    assert s['unassigned_fraction'] == pytest.approx(2 / 5)
    assert s['false_merge_rate'] == 0.0
    assert s['pair_precision'] == 1.0
    assert s['pair_recall'] == pytest.approx(1 / 4)  # 1 of 4 same-class pairs recovered
    assert s['false_split_rate'] == pytest.approx(3 / 4)


def test_all_singletons_has_undefined_precision_and_zero_recall() -> None:
    s = m.pairwise_scores(_arr(0, 0, 1), _arr(0, 1, 2))
    assert s['pair_precision'] is None
    assert s['false_merge_rate'] is None
    assert s['pair_recall'] == 0.0


def test_stratified_scores_split_the_items_by_stratum() -> None:
    truth = _arr(0, 0, 1, 1, 0, 0)
    cluster = _arr(0, 0, 1, 1, 2, 3)
    strata = np.array(['easy', 'easy', 'easy', 'easy', 'hard', 'hard'])
    out = m.stratified_scores(truth, cluster, strata)
    assert out['easy']['false_split_rate'] == 0.0
    assert out['hard']['false_split_rate'] == 1.0


def test_stability_ignores_label_names_and_flags_a_moved_item() -> None:
    a = _arr(0, 0, 1, 1, 2, 2)
    renamed = _arr(9, 9, 4, 4, 1, 1)
    same = m.assignment_stability(a, renamed)
    assert same['ari'] == 1.0
    assert same['identical_grouping'] is True
    moved = m.assignment_stability(a, _arr(9, 9, 4, 1, 1, 1))
    assert moved['ari'] < 1.0
    assert moved['identical_grouping'] is False


def test_stability_with_different_unassigned_items_is_not_identical() -> None:
    s = m.assignment_stability(_arr(0, 0, 1, 1), _arr(0, 0, 1, U))
    assert s['compared_items'] == 3
    assert s['identical_grouping'] is False


def test_moved_items_counts_only_protected_items_that_leave_their_cluster() -> None:
    reference = _arr(0, 0, 0, 1, 1, 2)
    candidate = _arr(5, 5, 6, 8, 8, U)  # item 2 left its cluster, item 5 lost its cluster
    protected = np.array([True, True, True, False, True, True])
    assert m.moved_items(reference, candidate, protected) == 2
    assert m.moved_items(reference, reference + 100, protected) == 0
    assert m.moved_items(reference, candidate, np.zeros(6, dtype=bool)) == 0


def _planted(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Truth with 200 classes; a partition with pure clusters, 5 mixed pairs, a 1 percent hole."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 200, n)
    cluster = truth * 2 + rng.integers(0, 2, n)  # every class split into two halves
    return truth, cluster


def test_planted_partition_at_scale_matches_hand_computed_counts_and_is_fast() -> None:
    n = 20_000
    truth, cluster = _planted(n, 1)
    # bridge: classes 0 and 1 merged into one cluster id, so exactly their halves join
    cluster = np.where(np.isin(cluster, [0, 2]), 0, cluster)
    started = time.perf_counter()
    s = m.pairwise_scores(truth, cluster)
    assert time.perf_counter() - started < 1.0

    bridge = np.isin(truth, [0, 1]) & np.isin(cluster, [0])
    a, b = int((truth[bridge] == 0).sum()), int((truth[bridge] == 1).sum())
    assert a + b == int(bridge.sum())
    fp = a * b
    assert s['false_merge_rate'] == pytest.approx(
        fp / sum(int(x) * (int(x) - 1) // 2 for x in np.bincount(cluster))
    )
    assert s['false_split_rate'] > 0.4  # every class is cut in two halves
    assert s['mixed_clusters'] == 1


def test_stability_and_moved_at_scale_are_fast() -> None:
    n = 20_000
    _, cluster = _planted(n, 2)
    rerun = cluster.copy()
    rerun[:50] = (rerun[:50] + 1) % 400  # 50 items change cluster between runs
    started = time.perf_counter()
    st = m.assignment_stability(cluster, rerun)
    protected = np.zeros(n, dtype=bool)
    protected[:100] = True
    moved = m.moved_items(cluster, rerun, protected)
    assert time.perf_counter() - started < 2.0
    assert st['identical_grouping'] is False
    assert 0.99 < st['ari'] < 1.0
    assert 0 < moved <= 50
