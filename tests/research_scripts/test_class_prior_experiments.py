"""Hand-computed cases for the representation, novelty, transfer, streaming and scale arms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scripts.research.cluster_diag import (
    class_prior as cp,
    cut_transfer as ct,
    novelty as nv,
    representation_arms as ra,
    scale as sc,
    streaming as st,
)
from scripts.research.cluster_diag.dataset import Bundle


def _unit_at(*degrees: float) -> np.ndarray:
    rad = np.radians(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


ALL4 = np.ones(4, dtype=bool)


# ---------------------------------------------------------------- representation arms


def test_the_raw_arm_equals_the_plain_within_key_clusterer() -> None:
    vectors = _unit_at(0, 10, 90, 5, 95)
    keys = ['a', 'a', 'a', 'b', 'b']
    got = ra.ArmClusterer(vectors, keys, 'raw').partition(0.05)
    assert got.tolist() == cp.WithinKeyClusterer(vectors, keys).partition(0.05).tolist()


def test_every_arm_keeps_the_two_tight_pairs_together_and_apart_from_each_other() -> None:
    vectors = _unit_at(0, 4, 90, 94, 0, 4, 90, 94)  # two copies, so a class has 8 crops
    keys = ['k'] * 8
    pool = np.random.default_rng(0).standard_normal((300, 2)).astype(np.float32)
    for arm in ra.ARMS:
        clusterer = ra.ArmClusterer(vectors, keys, arm, pool)
        top = np.sort(clusterer.heights())[-2:]  # the largest within-group merge, then the split
        part = clusterer.partition(float(top.mean()))
        assert part[0] == part[1] == part[4] == part[5], arm
        assert part[2] == part[3] == part[6] == part[7], arm
        assert part[0] != part[2], arm


def test_a_class_of_two_crops_falls_back_to_the_raw_distance() -> None:
    clusterer = ra.ArmClusterer(_unit_at(0, 10), ['k', 'k'], 'kreciprocal')
    # 1 - cos(10 deg) = 0.01519 is the only merge height
    assert clusterer.heights().tolist() == pytest.approx([1 - np.cos(np.radians(10))], abs=1e-6)


def test_the_whitening_arm_needs_a_fit_pool() -> None:
    with pytest.raises(ValueError, match='fit pool'):
        ra.distance_fn('pca_whiten_256', None)
    with pytest.raises(ValueError, match='unknown arm'):
        ra.distance_fn('nope', None)


def test_matched_cut_takes_the_candidate_closest_in_bcubed_recall() -> None:
    truth = np.array(['a', 'a', 'b', 'b'])
    alone = np.array([0, 1, 2, 3])  # BCubed recall: every crop finds 1 of 2 -> 0.5
    perfect = np.array([0, 0, 1, 1])  # BCubed recall 1.0
    cands = [(0.1, alone), (0.2, perfect)]
    assert ra.matched_cut(cands, truth, ALL4, 0.9) == 0.2
    assert ra.matched_cut(cands, truth, ALL4, 0.55) == 0.1


def test_an_arm_that_is_more_precise_at_equal_recall_passes() -> None:
    truth = np.array(['a', 'a', 'b', 'b'])
    merged = np.array([0, 0, 0, 0])  # BCubed precision 0.5, recall 1
    perfect = np.array([0, 0, 1, 1])  # precision 1, recall 1
    cell = ra.compare_at_matched_recall([(0.3, perfect)], merged, truth, {'s': ALL4}, ALL4, ALL4)
    assert cell['bcubed_precision_gain'] == pytest.approx(0.5)
    assert cell['bcubed_recall_gap'] == pytest.approx(0.0)
    assert cell['matched']
    assert cell['passes']
    assert cell['best_case_gain'] == pytest.approx(0.5)


def test_an_arm_that_loses_precision_fails_and_one_at_another_recall_is_unmatched() -> None:
    truth = np.array(['a', 'a', 'b', 'b'])
    merged = np.array([0, 0, 0, 0])
    perfect = np.array([0, 0, 1, 1])
    worse = ra.compare_at_matched_recall([(0.3, merged)], perfect, truth, {'s': ALL4}, ALL4, ALL4)
    assert worse['bcubed_precision_gain'] == pytest.approx(-0.5)
    assert worse['matched']
    assert not worse['passes']
    assert worse['worst_stratum'] == 'stratum:s'
    assert worse['worst_stratum_loss'] == pytest.approx(0.5)
    alone = np.array([0, 1, 2, 3])
    far = ra.compare_at_matched_recall([(0.3, alone)], perfect, truth, {'s': ALL4}, ALL4, ALL4)
    assert far['bcubed_recall_gap'] == pytest.approx(-0.5)
    assert not far['matched']
    assert not far['passes']
    assert far['best_case_gain'] is None  # no cut of the arm has a recall near raw's


def test_a_gain_just_under_two_hundredths_does_not_pass() -> None:
    # raw: cluster 0 = 50 a + 1 b, cluster 1 = 49 b. BCubed precision: 50 a at 50/51, the stray b
    # at 1/51, 49 b at 1 -> (50 * 50 / 51 + 1 / 51 + 49) / 100 = 0.98039; recall equals it
    truth = np.array(['a'] * 50 + ['b'] * 50)
    raw = np.array([0] * 51 + [1] * 49)
    arm = np.array([0] * 50 + [1] * 50)
    ones = np.ones(100, dtype=bool)
    cell = ra.compare_at_matched_recall([(0.3, arm)], raw, truth, {}, ones, ones)
    assert cell['raw']['bcubed_precision'] == pytest.approx((2500 / 51 + 1 / 51 + 49) / 100)
    assert cell['bcubed_precision_gain'] == pytest.approx(1 - 0.980392, abs=1e-5)
    assert cell['matched']  # the recall gap is the same 0.0196
    assert not cell['passes']  # 0.0196 < 0.02


def test_arm_verdict_distinguishes_pass_fail_and_inconclusive() -> None:
    ok = {'passes': True, 'matched': True, 'best_case_gain': 0.1}
    bad = {'passes': False, 'matched': True, 'best_case_gain': 0.0}
    unmatched = {'passes': False, 'matched': False, 'best_case_gain': 0.05}
    hopeless = {'passes': False, 'matched': False, 'best_case_gain': 0.019}
    no_cut = {'passes': False, 'matched': False, 'best_case_gain': None}
    assert ra.arm_verdict([ok, ok]) == 'PASS'
    assert ra.arm_verdict([ok, bad]) == 'FAIL'
    assert ra.arm_verdict([ok, unmatched]) == 'INCONCLUSIVE'
    assert ra.arm_verdict([ok, no_cut]) == 'INCONCLUSIVE'
    assert ra.arm_verdict([ok, hopeless]) == 'FAIL'
    assert ra.arm_verdict([bad, unmatched]) == 'FAIL'


# ---------------------------------------------------------------- novelty


def test_auroc_counts_ties_as_half() -> None:
    assert nv.auroc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert nv.auroc(np.array([1.0]), np.array([1.0])) == 0.5
    assert nv.auroc(np.array([2.0, 0.0]), np.array([1.0])) == 0.5  # one win, one loss
    assert nv.auroc(np.array([]), np.array([1.0])) is None


def test_crop_novelty_is_one_minus_the_mean_cosine_to_the_nearest_examples() -> None:
    # examples of key a at 0 and 60 degrees (images g0, g1); a query at 20 degrees: cosines
    # cos 20 = 0.93969 and cos 40 = 0.76604 -> mean 0.85287 over both -> novelty 0.14713
    vectors = _unit_at(0, 60, 20, 20)
    groups = ['g0', 'g1', 'gq', 'g0']
    got = nv.crop_novelty(vectors, ['a'] * 4, groups, np.array([0, 1]), ['a', 'a'], k=2)
    assert got[2] == pytest.approx(
        1 - (np.cos(np.radians(20)) + np.cos(np.radians(40))) / 2, abs=1e-6
    )
    # the last crop shares image g0 with the first example, which is then skipped: only the
    # 60-degree example remains -> novelty 1 - cos 40 = 0.23396
    assert got[3] == pytest.approx(1 - np.cos(np.radians(40)), abs=1e-6)


def test_crop_novelty_is_nan_without_an_example_of_the_key_or_any_other_image() -> None:
    vectors = _unit_at(0, 20)
    got = nv.crop_novelty(vectors, ['a', 'b'], ['g0', 'g1'], np.array([0]), ['a'], k=1)
    assert np.isnan(got[1])  # key b has no example
    alone = nv.crop_novelty(vectors, ['a', 'a'], ['g0', 'g0'], np.array([0]), ['a'], k=1)
    assert np.isnan(alone[1])  # the only example is from the query's own image


def test_cluster_novelty_averages_the_finite_member_scores() -> None:
    got = nv.cluster_novelty(np.array([0, 0, 1, 2]), np.array([0.2, 0.4, np.nan, 0.5]))
    assert got[:2].tolist() == pytest.approx([0.3, 0.3])
    assert np.isnan(got[2])
    assert got[3] == pytest.approx(0.5)


def test_thresholds_need_enough_own_class_crops_and_flags_need_a_threshold() -> None:
    keys = np.array(['a'] * 4 + ['b'])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    own = np.arange(5)
    th = nv.own_class_thresholds(keys, scores, own, 0.5, min_examples=2)
    assert th == {'a': pytest.approx(0.25)}  # median of 0.1 .. 0.4; key b has one crop only
    got = nv.flagged(keys, np.array([0.26, 0.24, np.nan, 0.5, 0.9]), th)
    assert got.tolist() == [True, False, False, True, False]  # b has no threshold, NaN never flags


def test_flag_quality_counts_hits() -> None:
    q = nv.flag_quality(np.array([True, True, True, False]), np.array([True, True, False, True]))
    assert q['precision'] == pytest.approx(2 / 3)
    assert q['recall'] == pytest.approx(2 / 3)
    assert nv.flag_quality(np.zeros(2, dtype=bool), np.ones(2, dtype=bool))['precision'] is None


def test_choose_quantile_takes_the_lowest_one_that_reaches_the_precision_bar() -> None:
    # 30 own-class crops scored 0 .. 29 and 10 open-set crops scored 100: any q flags exactly
    # the open-set crops once the threshold is above the own-class maximum is not true at low
    # q, so the own-class crops above the median are flagged too and spoil the precision.
    keys = np.array(['a'] * 40)
    scores = np.array([float(i) for i in range(30)] + [100.0] * 10)
    own = np.arange(30)
    novel = np.array([False] * 30 + [True] * 10)
    q, table = nv.choose_quantile(keys, scores, own, np.arange(40), novel, bar=0.8)
    # q 0.5 -> threshold 14.5: flags 15 own + 10 open -> precision 0.4; q 0.75 -> 21.75: 8 + 10 -> 0.556;
    # q 0.9 -> 26.1: 3 + 10 -> 0.769; q 0.95 -> 27.55: 2 + 10 -> 0.833 -> chosen
    assert [round(r['precision'], 3) for r in table[:4]] == [0.4, 0.556, 0.769, 0.833]
    assert q == 0.95


def test_per_key_auroc_judges_only_keys_with_enough_examples_and_novel_crops() -> None:
    keys = np.array(['a'] * 6)
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.2])
    truth = np.array(['a', 'a', 'a', 'lion', 'lion', 'a'])
    got = nv.per_key_auroc(keys, scores, np.arange(6), truth, {'a'}, {'a': 3})
    assert got['a']['auroc'] == 1.0
    assert got['a']['own_class_scored'] == 4
    assert got['a']['novel_scored'] == 2
    assert got['a']['judged'] is False  # 3 examples < 20 and 2 novel < 10


def test_threshold_for_precision_takes_the_last_rank_that_keeps_the_bar() -> None:
    ranked = [(0.9, True), (0.8, True), (0.7, False), (0.6, True)]
    # after 2 suggestions precision is 1.0; after 3, 2/3; after 4, 3/4 - only rank 2 qualifies
    assert nv.threshold_for_precision(ranked, 0.9, min_count=2) == 0.8
    assert nv.threshold_for_precision(ranked, 0.9, min_count=3) is None


def test_suggestions_pair_flagged_clusters_of_different_keys_by_best_member_pair() -> None:
    # cluster 0 (key a): 0 and 10 degrees; cluster 1 (key b): 15 and 90 degrees; cluster 2
    # (key a): 100 degrees. Best pair of 0 and 1 is 10 vs 15 -> cos 5; 1 and 2 is 90 vs 100 -> cos 10;
    # clusters 0 and 2 share a key -> no suggestion.
    vectors = _unit_at(0, 10, 15, 90, 100)
    part = np.array([0, 0, 1, 1, 2])
    keys = ['a', 'a', 'b', 'b', 'a']
    got = nv.suggestions(vectors, part, keys, [0, 1, 2])
    assert [(a, b) for _, a, b in got] == [(0, 1), (1, 2)]  # best first: cos 5 > cos 10
    assert got[0][0] == pytest.approx(np.cos(np.radians(5)), abs=1e-6)
    assert got[1][0] == pytest.approx(np.cos(np.radians(10)), abs=1e-6)


def test_merge_clusters_joins_the_suggested_pairs_transitively() -> None:
    part = np.array([0, 0, 1, 2, 3])
    merged = nv.merge_clusters(part, [(0.9, 0, 1), (0.8, 1, 2)])
    assert merged[0] == merged[1] == merged[2] == merged[3]
    assert merged[4] != merged[0]
    assert nv.merge_clusters(part, []).tolist() == part.tolist()


def test_suggestion_correct_needs_matching_open_set_majorities() -> None:
    part = np.array([0, 0, 1, 1, 2])
    truth = np.array(['lion', 'cat', 'lion', 'lion', 'tiger'])
    rows = np.arange(5)
    judged = np.ones(5, dtype=bool)
    keys = {'cat'}
    # cluster 0 has one lion and one cat -> open-set majority lion; cluster 1 lion; cluster 2 tiger
    assert nv.suggestion_correct((1.0, 0, 1), part, truth, judged, rows, keys) is True
    assert nv.suggestion_correct((1.0, 0, 2), part, truth, judged, rows, keys) is False
    only_cat = np.array(['cat'] * 5)
    assert nv.suggestion_correct((1.0, 0, 1), part, only_cat, judged, rows, keys) is None


def test_open_set_pair_recall_looks_only_at_open_set_labels() -> None:
    truth = np.array(['lion', 'lion', 'lion', 'cat', 'cat'])
    part = np.array([0, 0, 1, 2, 3])
    got = nv.open_set_pair_recall(truth, part, np.ones(5, dtype=bool), {'cat'})
    assert got == pytest.approx(1 / 3)  # 3 lion pairs, one kept together
    assert (
        nv.open_set_pair_recall(truth, part, np.array([0, 0, 0, 1, 1], dtype=bool), {'cat'}) is None
    )


def _novelty_bundle() -> tuple[Bundle, np.ndarray, np.ndarray]:
    """Cats near e0, lions (filed under cat) near e1, dogs near e2; one image per crop."""
    rng = np.random.default_rng(1)
    centres = {'cat': 0, 'lion': 1, 'dog': 2}
    plan = [('cat', 'cat')] * 50 + [('lion', 'cat')] * 24 + [('dog', 'dog')] * 50
    vecs = np.stack([np.eye(8)[centres[t]] + 0.05 * rng.standard_normal(8) for t, _ in plan])
    vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)
    n = len(plan)
    meta = [
        {'crop_id': f'c{i:03d}', 'image_path': f'/x/{i}.jpg', 'detector_class_name': d}
        for i, (_, d) in enumerate(plan)
    ]
    b = Bundle(
        vectors=vecs,
        meta=meta,
        labelled=np.arange(n, dtype=np.int64),
        truth=np.array([t for t, _ in plan]),
        stratum=np.array(['rare'] * n),
        group=[f'{i}.jpg' for i in range(n)],
        aspect=np.ones(n),
        side=np.full(n, 100.0),
        protected=np.zeros(n, dtype=bool),
    )
    part = np.array([0] * 50 + [1] * 24 + [2] * 50)
    first = np.arange(n) % 2 == 0
    return b, part, first


def test_run_fold_finds_the_open_set_cluster_filed_under_the_wrong_class() -> None:
    b, part, first = _novelty_bundle()
    keys = np.array(['cat'] * 74 + ['dog'] * 50)
    out = nv.run_fold(b, keys, part, first, ~first)
    cat = out['per_key_auroc_cluster_score']['cat']
    assert cat['auroc'] == 1.0  # the lion cluster is far from every validated cat
    assert cat['judged']  # 25 examples, 12 open-set crops in the scored half
    assert out['auroc_bar'] == {'judged_keys': 1, 'min_auroc': 1.0, 'bar': 0.9, 'pass': True}
    assert out['suggestion_basis'] == 'calibrated_quantile'
    assert out['flag_quality_scored_by_quantile']['0.5']['precision'] == 1.0
    assert out['chosen_quantile'] == 0.5
    # only the lion cluster is flagged: 12 scored lions, no cat or dog
    assert out['flag_quality_scored']['flagged'] == 12
    assert out['flag_quality_scored']['precision'] == 1.0
    assert out['flag_quality_scored']['recall'] == 1.0
    assert out['open_set_crops_scored'] == 12


# ---------------------------------------------------------------- cut transfer


def test_subsample_per_key_is_seeded_and_stays_inside_the_eligible_crops() -> None:
    keys = np.array(['a', 'a', 'a', 'b', 'b', 'c'])
    eligible = np.array([True, True, True, True, False, False])
    got = ct.subsample_per_key(keys, eligible, 2, seed=3)
    assert got.tolist() == ct.subsample_per_key(keys, eligible, 2, seed=3).tolist()
    assert not got[~eligible].any()
    assert got[:3].sum() == 2  # two of key a
    assert got[3]  # the only eligible b
    assert any(
        ct.subsample_per_key(keys, eligible, 2, seed=s).tolist() != got.tolist() for s in range(10)
    )


def test_sample_size_table_finds_the_smallest_count_that_reproduces_the_cut() -> None:
    truth = np.array(['a', 'a', 'b', 'b'] * 4)
    keys = np.array(['k1'] * 4 + ['k2'] * 4 + ['k1'] * 4 + ['k2'] * 4)
    perfect = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    merged = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    first = np.array([True] * 8 + [False] * 8)
    table = ct.sample_size_table(
        [(0.2, perfect), (0.4, merged)],
        truth,
        first,
        0.1,
        lambda eligible, n, seed: ct.subsample_per_key(keys, eligible, n, seed),
        (1, 4),
    )
    # one crop per key leaves no same-key pair, so no cut can be chosen; four per key is the
    # whole calibration half and gives the full cut 0.2 every time
    assert table['full_calibration_cut'] == 0.2
    assert table['counts']['1']['within_tolerance'] == 0
    assert table['counts']['4']['within_tolerance'] == 10
    assert table['smallest_count_within_tolerance'] == 4


def test_subsample_total_keeps_the_requested_number_of_eligible_crops() -> None:
    eligible = np.array([True] * 6 + [False] * 4)
    got = ct.subsample_total(eligible, 3, seed=1)
    assert got.sum() == 3
    assert not got[~eligible].any()
    assert got.tolist() == ct.subsample_total(eligible, 3, seed=1).tolist()
    assert ct.subsample_total(eligible, 50, seed=1).sum() == 6


def test_transfer_report_scores_the_cut_on_the_second_project() -> None:
    vectors = _unit_at(0, 3, 90, 93)
    truth = np.array(['a', 'a', 'b', 'b'])
    out = ct.transfer_report(vectors, ['k', 'k', 'k', 'k'], truth, 0.1, ALL4, {}, np.arange(4))
    # 1 - cos(3) = 0.0014 merges within each pair; 1 - cos(87) = 0.95 keeps the pairs apart
    assert out['views']['all']['bcubed_precision'] == pytest.approx(1.0)
    assert out['views']['all']['bcubed_recall'] == pytest.approx(1.0)
    assert out['detector_class_only']['bcubed_precision'] == pytest.approx(0.5)


# ---------------------------------------------------------------- streaming


def test_late_groups_are_a_seeded_share_of_the_images() -> None:
    groups = [f'g{i}' for i in range(10)]
    got = st.late_groups(groups, 0.2, 5)
    assert len(got) == 2
    assert got == st.late_groups(reversed(groups), 0.2, 5)
    assert any(got != st.late_groups(groups, 0.2, s) for s in range(6, 12))


def test_late_images_are_not_the_images_the_split_ranks_first() -> None:
    import hashlib

    groups = [f'g{i}' for i in range(200)]
    by_split = sorted(groups, key=lambda g: hashlib.sha256(f'7:{g}'.encode()).hexdigest())
    # the evaluation split ranks images by sha256 of "seed:image"; late arrivals use their own
    # namespace, so they are not simply the images at the head of that ranking
    assert st.late_groups(groups, 0.2, 7) != set(by_split[:40])


def test_cluster_distance_by_method() -> None:
    members = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    x = np.array([1.0, 0.0], dtype=np.float32)
    assert st.cluster_distances(x, members, 'best_member') == pytest.approx(0.0)
    assert st.cluster_distances(x, members, 'mean_member') == pytest.approx(0.5)
    # centroid (0.5, 0.5) has norm 0.7071; cosine 0.5 / 0.7071 = 0.7071 -> distance 0.2929
    assert st.cluster_distances(x, members, 'centroid') == pytest.approx(1 - 0.5 / np.sqrt(0.5))


def test_assign_abstains_beyond_the_cut_or_inside_the_margin() -> None:
    near = _unit_at(0, 2)
    far = _unit_at(40, 42)
    clusters = {0: near, 1: far}
    x = _unit_at(1)[0]
    assert st.assign(x, clusters, 'best_member', 0.1, 0.0) == 0
    assert st.assign(x, {1: far}, 'best_member', 0.1, 0.0) is None  # nearest is beyond the cut
    assert st.assign(x, clusters, 'best_member', 0.1, 0.5) is None  # second is not far enough
    assert st.assign(x, {0: near}, 'best_member', 0.1, 0.5) == 0  # a lone cluster has no margin
    assert st.assign(x, {}, 'best_member', 0.1, 0.0) is None


def test_plurality_breaks_ties_alphabetically() -> None:
    assert st.plurality(['b', 'a', 'b']) == 'b'
    assert st.plurality(['b', 'a']) == 'a'
    assert st.plurality([]) is None


def test_score_arrivals_counts_only_judgeable_accepted_assignments() -> None:
    assigned = {1: 0, 2: 0, 3: None, 4: 1}
    truth_of = {1: 'a', 2: 'b', 3: 'a', 4: 'z'}
    got = st.score_arrivals(assigned, truth_of, {0: 'a', 1: None}, nameable={1})
    assert got['accepted'] == 3
    assert got['abstain_share'] == pytest.approx(0.25)
    assert got['judged'] == 2  # cluster 1 has no labelled member
    assert got['label_precision'] == pytest.approx(0.5)  # arrival 1 matches 'a', arrival 2 is 'b'
    assert got['judged_nameable'] == 1
    assert got['label_precision_nameable'] == pytest.approx(1.0)


def test_choose_margin_takes_the_smallest_that_reaches_the_bar_on_enough_arrivals() -> None:
    table: dict[float, dict[str, Any]] = {
        0.0: {'judged': 30, 'label_precision': 0.90},
        0.05: {'judged': 4, 'label_precision': 1.0},
        0.1: {'judged': 20, 'label_precision': 0.96},
    }
    assert st.choose_margin(table) == 0.1
    assert st.choose_margin({0.0: {'judged': 30, 'label_precision': 0.9}}) is None


def test_incremental_against_refit_counts_an_abstained_arrival_as_a_singleton() -> None:
    base_part = np.array([0, 0, 1, -1])
    base = np.array([True, True, True, False])
    refit = np.array([0, 0, 1, 1])
    joined = st.incremental_vs_refit(base_part, {3: 1}, refit, base)
    assert joined['ari_all_crops'] == pytest.approx(1.0)
    left = st.incremental_vs_refit(base_part, {3: None}, refit, base)
    assert left['ari_all_crops'] < 1.0
    assert left['ari_base_crops'] == pytest.approx(1.0)


def test_streaming_run_assigns_arrivals_to_their_own_class_cluster() -> None:
    rng = np.random.default_rng(2)
    n_per = 100
    keys = np.array(['a'] * n_per + ['b'] * n_per + ['c'] * n_per)
    centres = np.eye(8)
    vectors = np.stack(
        [centres[i // n_per] + 0.05 * rng.standard_normal(8) for i in range(3 * n_per)]
    )
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
    groups = [f'g{i}' for i in range(3 * n_per)]
    first = {g for i, g in enumerate(groups) if i % 2 == 0}
    out = st.run_streaming(
        vectors, keys, groups, np.arange(3 * n_per), keys, first, cut=0.3, seed=1
    )
    assert out['arrivals'] == 60
    for method in st.METHODS:
        fold = out['methods'][method]['folds']['fold1_first_to_second']
        assert fold['margin'] == 0.0
        assert fold['scored']['label_precision'] == 1.0
        assert fold['scored']['abstain_share'] == 0.0
        assert fold['incremental_vs_refit']['ari_all_crops'] == pytest.approx(1.0)


# ---------------------------------------------------------------- scale


def test_condensed_distance_orders_the_upper_triangle_row_major() -> None:
    v = _unit_at(0, 90, 45)
    got = sc.condensed_cosine_distance(v, chunk=2)
    # pairs (0,1), (0,2), (1,2): cos 90 = 0, cos 45 = 0.7071, cos 45 = 0.7071
    assert got.tolist() == pytest.approx(
        [1.0, 1 - np.cos(np.pi / 4), 1 - np.cos(np.pi / 4)], abs=1e-6
    )


def test_identical_rows_never_give_a_negative_distance() -> None:
    v = np.repeat(_unit_at(37)[:1], 3, axis=0)  # cos(37 deg)^2 + sin^2 rounds off 1 in float32
    assert (sc.condensed_cosine_distance(v) >= 0).all()
    assert sc.dense_average_partition(v, 0.01).tolist() == [0, 0, 0]


def test_dense_partition_cuts_at_the_height_and_handles_one_crop() -> None:
    v = _unit_at(0, 1, 90, 91)
    assert sc.dense_average_partition(v, 0.05).tolist() in ([0, 0, 1, 1], [1, 1, 0, 0])
    assert sc.dense_average_partition(v[:1], 0.05).tolist() == [0]
    assert len(set(sc.dense_average_partition(v, 1e-9).tolist())) == 4


def test_mutual_knn_graph_keeps_only_reciprocal_pairs() -> None:
    v = _unit_at(0, 1, 90, 91)
    adj = sc.mutual_knn_graph(v, k=1)
    assert adj.toarray().tolist() == [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]


def test_sparse_partition_matches_dense_on_separated_groups() -> None:
    v = _unit_at(0, 1, 2, 90, 91, 92)
    sparse = sc.sparse_average_partition(v, 0.05, k=2)
    dense = sc.dense_average_partition(v, 0.05)
    assert len(set(sparse.tolist())) == len(set(dense.tolist())) == 2
    assert sparse[0] == sparse[1] == sparse[2] != sparse[3]
    assert sparse[3] == sparse[4] == sparse[5]


def test_within_key_codes_are_unique_across_keys() -> None:
    v = _unit_at(0, 1, 0, 1)
    out = sc.within_key(v, ['a', 'a', 'b', 'b'], 0.05, sc.dense_average_partition)
    assert out[0] == out[1]
    assert out[2] == out[3]
    assert out[0] != out[2]
