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


# --- BCubed, AMI, risk-coverage and attribute pair strata ---


def _bcubed_reference(truth: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """Definition-level O(n^2) BCubed; an unassigned item is its own cluster."""
    n = len(truth)
    cid = [(-1 - i) if c == U else int(c) for i, c in enumerate(cluster)]
    prec = rec = 0.0
    for i in range(n):
        same_c = [j for j in range(n) if cid[j] == cid[i]]
        same_t = [j for j in range(n) if truth[j] == truth[i]]
        both = [j for j in same_c if truth[j] == truth[i]]
        prec += len(both) / len(same_c)
        rec += len(both) / len(same_t)
    return prec / n, rec / n


def test_bcubed_perfect_partition_is_one() -> None:
    s = m.bcubed_scores(_arr(0, 0, 1, 1, 2), _arr(5, 5, 6, 6, 7))
    assert (s['bcubed_precision'], s['bcubed_recall'], s['bcubed_f']) == (1.0, 1.0, 1.0)


def test_bcubed_all_in_one_cluster_keeps_recall_and_loses_precision() -> None:
    s = m.bcubed_scores(_arr(0, 0, 1, 1), _arr(0, 0, 0, 0))
    assert s['bcubed_precision'] == pytest.approx(0.5)
    assert s['bcubed_recall'] == 1.0
    assert s['bcubed_f'] == pytest.approx(2 * 0.5 / 1.5)


def test_bcubed_all_singletons_keeps_precision_and_loses_recall() -> None:
    s = m.bcubed_scores(_arr(0, 0, 1, 1), _arr(0, 1, 2, 3))
    assert s['bcubed_precision'] == 1.0
    assert s['bcubed_recall'] == pytest.approx(0.5)


def test_bcubed_hand_computed_mixed_example() -> None:
    # truth A A A B B ; clusters {A A B} {A} {B}
    # precision per item: 2/3 2/3 1 (A alone) ... item order A1 A2 B1 | A3 | B2
    #   A1 A2 in cluster0 with B1: 2/3, 2/3 ; B1: 1/3 ; A3: 1 ; B2: 1  -> (2/3+2/3+1/3+1+1)/5 = 11/15
    # recall per item: A1 A2: 2/3 each ; A3: 1/3 ; B1: 1/2 ; B2: 1/2 -> (2/3+2/3+1/3+1/2+1/2)/5 = 8/15
    s = m.bcubed_scores(_arr(0, 0, 1, 0, 1), _arr(0, 0, 0, 1, 2))
    assert s['bcubed_precision'] == pytest.approx(11 / 15)
    assert s['bcubed_recall'] == pytest.approx(8 / 15)


def test_bcubed_matches_the_definition_on_random_partitions_with_holes() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        truth = rng.integers(0, 5, 40)
        cluster = rng.integers(0, 6, 40)
        cluster[rng.random(40) < 0.15] = U
        p, r = _bcubed_reference(truth, cluster)
        s = m.bcubed_scores(truth, cluster)
        assert s['bcubed_precision'] == pytest.approx(p)
        assert s['bcubed_recall'] == pytest.approx(r)


def test_ami_matches_scikit_learn() -> None:
    from sklearn.metrics import adjusted_mutual_info_score

    rng = np.random.default_rng(4)
    for n, k_t, k_c in [(60, 4, 5), (300, 10, 40), (500, 25, 12)]:
        truth = rng.integers(0, k_t, n)
        noisy = np.where(rng.random(n) < 0.6, truth, rng.integers(0, k_c, n))
        for cluster in (noisy, rng.integers(0, k_c, n)):
            assert m.adjusted_mutual_info(truth, cluster) == pytest.approx(
                adjusted_mutual_info_score(truth, cluster), abs=1e-9
            )


def test_ami_edges() -> None:
    assert m.adjusted_mutual_info(_arr(0, 0, 1, 1), _arr(9, 9, 4, 4)) == pytest.approx(1.0)
    assert m.adjusted_mutual_info(_arr(0, 0, 0, 0), _arr(1, 1, 1, 1)) == 1.0


def test_ami_counts_unassigned_items_as_singletons() -> None:
    from sklearn.metrics import adjusted_mutual_info_score

    truth = _arr(0, 0, 1, 1, 2, 2)
    cluster = _arr(0, 0, U, U, 1, 1)
    as_singletons = _arr(0, 0, 10, 11, 1, 1)
    assert m.adjusted_mutual_info(truth, cluster) == pytest.approx(
        adjusted_mutual_info_score(truth, as_singletons), abs=1e-9
    )


def test_risk_coverage_without_confidence_is_one_point() -> None:
    truth = _arr(0, 0, 0, 1, 1)
    cluster = _arr(0, 0, 0, 0, U)  # item 3 is a minority member of cluster 0, item 4 abstained
    rc = m.risk_coverage(truth, cluster)
    assert rc['coverage'] == pytest.approx(4 / 5)
    assert rc['risk'] == pytest.approx(1 / 4)
    assert rc['abstained'] == 1
    assert rc['curve'] is None


def test_risk_coverage_curve_orders_by_confidence() -> None:
    truth = _arr(0, 0, 0, 0)
    cluster = _arr(0, 0, 0, 1)  # one item per cluster 1: all pure, so shape the error by truth
    truth = _arr(0, 0, 1, 0)  # item 2 is the minority of cluster 0
    conf = np.array([0.9, 0.8, 0.1, 0.7])
    rc = m.risk_coverage(truth, cluster, conf)
    # accepted in order: item0 (ok), item1 (ok), item3 (ok), item2 (wrong)
    assert rc['curve'] == [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.25)]
    assert rc['aurc'] == pytest.approx((0 + 0 + 0 + 0.25) / 4)


def test_risk_coverage_never_covers_abstained_items() -> None:
    truth = _arr(0, 0, 1)
    cluster = _arr(0, U, U)
    rc = m.risk_coverage(truth, cluster, np.array([0.9, 0.99, 0.99]))
    assert rc['curve'] == [(pytest.approx(1 / 3), 0.0)]
    assert rc['coverage'] == pytest.approx(1 / 3)


def test_attribute_pair_scores_split_pairs_by_same_and_different_view() -> None:
    # class 0: front front side ; class 1: front side. cluster: all of class 0 together,
    # class 1 split.  same-class pairs: (0,1) same view, (0,2) (1,2) different, (3,4) different
    truth = _arr(0, 0, 0, 1, 1)
    cluster = _arr(0, 0, 0, 1, 2)
    view = np.array(['front', 'front', 'side', 'front', 'side'])
    out = m.attribute_pair_scores(truth, cluster, view)
    assert out['same']['same_class_pairs'] == 1
    assert out['same']['pair_recall'] == 1.0
    assert out['different']['same_class_pairs'] == 3
    assert out['different']['pair_recall'] == pytest.approx(2 / 3)
    assert out['different']['false_merge_rate'] == 0.0


def test_attribute_pair_scores_count_false_merges_by_attribute() -> None:
    truth = _arr(0, 1, 0, 1)
    cluster = _arr(0, 0, 1, 1)  # both clusters mix the classes
    size = np.array(['big', 'big', 'small', 'big'])
    out = m.attribute_pair_scores(truth, cluster, size)
    assert out['same']['false_merge_rate'] == 1.0  # (0,1) big/big, merged wrongly
    assert out['different']['false_merge_rate'] == 1.0  # (2,3) small/big, merged wrongly
    assert out['same']['coclustered_pairs'] == 1
    assert out['different']['coclustered_pairs'] == 1


def test_attribute_pair_scores_match_brute_force() -> None:
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 4, 60)
    cluster = rng.integers(0, 6, 60)
    attr = rng.integers(0, 3, 60)
    out = m.attribute_pair_scores(truth, cluster, attr)
    for key, same in (('same', True), ('different', False)):
        sc = rec = co = fm = 0
        for i in range(60):
            for j in range(i + 1, 60):
                if (attr[i] == attr[j]) != same:
                    continue
                st, sc_ = truth[i] == truth[j], cluster[i] == cluster[j]
                sc += st
                rec += st and sc_
                co += sc_
                fm += sc_ and not st
        assert out[key]['same_class_pairs'] == sc
        assert out[key]['coclustered_pairs'] == co
        assert out[key]['pair_recall'] == pytest.approx(rec / sc)
        assert out[key]['false_merge_rate'] == pytest.approx(fm / co)
