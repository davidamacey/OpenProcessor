"""Hand-computed cases for the clustering diagnostics (scripts/research/cluster_diag).

Every expected value below is worked out on paper in the comment beside it, not read off the
function under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.bench.cluster_eval_metrics import UNASSIGNED
from scripts.research.cluster_diag import diagnostics as dg, partitions as pt


def _unit_at(*degrees: float) -> np.ndarray:
    rad = np.radians(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


# A 4-point similarity: 0-1 and 2-3 are close pairs, 3 is also fairly close to 1.
SIM4 = np.array(
    [
        [1.0, 0.9, 0.1, 0.2],
        [0.9, 1.0, 0.3, 0.4],
        [0.1, 0.3, 1.0, 0.8],
        [0.2, 0.4, 0.8, 1.0],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------- geometry


def test_l2_normalize_gives_unit_rows_and_keeps_a_zero_row() -> None:
    out = dg.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_effective_rank_counts_the_directions_that_carry_variance() -> None:
    # +-e_i over four axes: the covariance is I/4 on four axes, a uniform spectrum -> rank 4
    four = np.vstack([np.eye(4), -np.eye(4)]).astype(np.float32)
    assert dg.effective_rank(four) == pytest.approx(4.0, rel=1e-4)
    # +-e_0 only: all variance on one axis -> rank 1
    one = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=np.float32)
    assert dg.effective_rank(one) == pytest.approx(1.0, rel=1e-4)
    # spectrum (3/4, 1/4): exp(-(.75 ln .75 + .25 ln .25)) = exp(0.5623) = 1.7548
    skewed = np.array([[3, 0], [-3, 0], [0, math.sqrt(3)], [0, -math.sqrt(3)]], dtype=np.float32)
    assert dg.effective_rank(skewed) == pytest.approx(1.7548, abs=1e-3)


def test_top_k_neighbours_orders_best_first_and_excludes_self() -> None:
    # row 0: 1 (.9) then 3 (.2); row 2: 3 (.8) then 1 (.3)
    got = dg.top_k_neighbours(SIM4, 2)
    assert got[0].tolist() == [1, 3]
    assert got[1].tolist() == [0, 3]
    assert got[2].tolist() == [3, 1]
    assert got[3].tolist() == [2, 1]


def test_k_occurrence_counts_how_often_a_row_is_someones_neighbour() -> None:
    # k = 2 lists: 0:[1,3] 1:[0,3] 2:[3,1] 3:[2,1] -> row 1 appears 3 times, 3 appears 3 times
    assert dg.k_occurrence(dg.top_k_neighbours(SIM4, 2), 4).tolist() == [1, 3, 1, 3]


def test_hubness_of_a_star_matches_the_hand_computed_skew() -> None:
    sim = np.full((5, 5), 0.1, dtype=np.float32)
    sim[0, 1:] = sim[1:, 0] = 0.9
    sim[0, 1] = sim[1, 0] = 0.95
    np.fill_diagonal(sim, 1.0)
    # k = 1: rows 1..4 point to 0, row 0 points to 1 -> N1 = [4, 1, 0, 0, 0]
    # mean 1, deviations [3, 0, -1, -1, -1]: m2 = 12/5, m3 = 24/5, skew = 4.8 / 2.4**1.5
    out = dg.hubness(sim, 1)
    assert out['max_N1'] == 4
    assert out['antihub_frac_N1'] == pytest.approx(3 / 5)
    assert out['hub_skew_N1'] == pytest.approx(4.8 / 2.4**1.5, rel=1e-6)


def test_geometry_reports_anisotropy_of_a_shifted_cloud() -> None:
    rng = np.random.default_rng(0)
    cloud = dg.l2_normalize(rng.standard_normal((300, 8)) + 5.0 * np.eye(8)[0])
    spread = dg.l2_normalize(rng.standard_normal((300, 8)))
    shifted, centred = dg.geometry(cloud, k=5), dg.geometry(spread, k=5)
    assert shifted['mean_random_cos'] > 0.7  # E[cos] ~ 25 / (25 + 8): a shared direction
    assert abs(centred['mean_random_cos']) < 0.05
    assert shifted['norm_of_mean'] > 0.8
    assert 1.0 <= centred['effective_rank'] <= 8.0


# ---------------------------------------------------------------- neighbour purity


def test_knn_report_on_two_pairs_of_classes() -> None:
    x = _unit_at(0, 10, 90, 100)
    sim = x @ x.T
    truth = ['a', 'a', 'b', 'b']
    # Nearest other point is always the partner (P@1 = 1); the second nearest is always the
    # other class (80 degrees away beats 90): P@2 = 1/2. Tie in the vote goes to the nearest.
    report = dg.knn_report(sim, truth, {'first_pair': np.array([True, True, False, False])}, k=2)
    assert report['P@1'] == 1.0
    assert report['P@2'] == 0.5
    assert report['2NN_vote_acc'] == 1.0
    assert report['P@2[first_pair]'] == 0.5


def test_knn_vote_accuracy_counts_a_wrong_majority_as_wrong() -> None:
    # point 0 (class a) sits among three class-b points: 0 is outvoted, the b points are fine
    x = _unit_at(0, 20, 25, 30, 180)
    truth = ['a', 'b', 'b', 'b', 'a']
    # k=3 votes: 0 -> b b b (wrong); 1 -> {0:a, 2:b, 3:b} b (right); 2 -> b (right);
    # 3 -> {2:b, 1:b, 0:a} b (right); 4 (180 deg) -> nearest by cos: 0 (a), then 1, 2: b b -> b (wrong)
    assert dg.knn_vote_accuracy(x @ x.T, truth, k=3) == pytest.approx(3 / 5)


# ---------------------------------------------------------------- transforms


def test_pca_whiten_gives_unit_variance_on_the_fit_pool() -> None:
    pool = np.array([[2, 1], [2, -1], [-2, 1], [-2, -1]], dtype=np.float32)  # variances 4 and 1
    mu, proj = dg.pca_whiten_fit(pool, dim=2, power=0.5)
    z = (pool - mu) @ proj
    assert np.allclose(z.T @ z / len(z), np.eye(2), atol=1e-4)


def test_pca_without_whitening_keeps_the_top_axis_only() -> None:
    pool = np.array([[2, 1], [2, -1], [-2, 1], [-2, -1]], dtype=np.float32)
    _, proj = dg.pca_whiten_fit(pool, dim=1, power=0.0)
    assert proj.shape == (2, 1)
    assert abs(proj[0, 0]) == pytest.approx(1.0, abs=1e-5)  # the x axis has variance 4
    assert abs(proj[1, 0]) < 1e-5


def test_pca_whiten_apply_returns_unit_rows() -> None:
    pool = np.array([[2, 1], [2, -1], [-2, 1], [-2, -1]], dtype=np.float32)
    mu, proj = dg.pca_whiten_fit(pool, dim=2)
    out = dg.pca_whiten_apply(np.array([[1.0, 0.5], [-3.0, 2.0]], dtype=np.float32), mu, proj)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_csls_matches_the_hand_computed_values() -> None:
    v = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    ref = np.array([[1, 0], [0, 1]], dtype=np.float32)
    # k = 1: r = best cosine to the reference = [1, 1, 0.8]
    got = dg.csls_similarity(v, ref, k=1)
    assert got[0, 1] == pytest.approx(0 - 1 - 1)
    assert got[0, 2] == pytest.approx(2 * 0.6 - 1 - 0.8)
    assert got[1, 2] == pytest.approx(2 * 0.8 - 1 - 0.8)
    assert got[2, 2] == pytest.approx(2 - 1.6)


def test_dba_matches_the_hand_computed_expansion() -> None:
    v = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    # k=2, alpha=3. Row 0 expands with itself (w 1) and row 2 (cos .6, w .216):
    #   (1.1296, .1728) / 1.14274 = (.98848, .15122)
    # Row 2 with itself (w 1) and row 1 (cos .8, w .512): (.6, 1.312) / 1.44269 = (.41589, .90945)
    # cosine of the expanded rows 0 and 2: .98848*.41589 + .15122*.90945 = .5486
    got = dg.dba_similarity(v, k=2, alpha=3.0)
    assert got[0, 2] == pytest.approx(0.5486, abs=1e-3)
    assert got[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_kreciprocal_on_two_isolated_pairs_matches_the_closed_form() -> None:
    v = _unit_at(0, 10, 100, 110)
    # k1 = 1, k2 = 1, lambda = 0 leaves only the Jaccard term. Each point's reciprocal set is
    # itself plus its partner, weights (1, e^-d)/(1 + e^-d) with d = 2 - 2 cos 10 deg = .030384.
    # Partner rows overlap by 2 e^-d / (1 + e^-d) = .98482; Jaccard distance = 1 - o / (2 - o).
    d = 2 - 2 * math.cos(math.radians(10))
    overlap = 2 * math.exp(-d) / (1 + math.exp(-d))
    got = dg.kreciprocal_similarity(v, k1=1, k2=1, lam=0.0)
    assert got[0, 1] == pytest.approx(-(1 - overlap / (2 - overlap)), abs=1e-5)
    assert got[0, 0] == pytest.approx(0.0, abs=1e-6)  # identical rows: no distance
    assert got[0, 2] == pytest.approx(-1.0, abs=1e-6)  # disjoint support: maximal distance
    assert np.allclose(got, got.T, atol=1e-6)


# ---------------------------------------------------------------- pair scores and curves


def test_score_partition_counts_pairs_on_the_labelled_rows() -> None:
    # partition [0,0,1,1] against classes a,a,a,b: same-cluster pairs (0,1),(2,3), of which only
    # (0,1) shares a class -> precision 1/2; same-class pairs aaa = 3 -> recall 1/3
    _, precision, recall, unassigned = dg.score_partition(
        np.array([0, 0, 1, 1]), ['a', 'a', 'a', 'b'], np.arange(4)
    )
    assert (precision, recall, unassigned) == (0.5, pytest.approx(1 / 3), 0.0)


def test_score_partition_reads_only_the_labelled_rows_and_counts_unassigned() -> None:
    partition = np.array([0, 0, 9, UNASSIGNED, 1, 1])
    labelled = np.array([0, 1, 3, 4, 5])  # row 2 is unlabelled and must not count
    # truth a, a, b, b, b: rows 0,1 -> cluster 0 (a,a); row 3 unassigned; rows 4,5 -> cluster 1
    # (b,b). Pairs in clusters: 2, both correct -> precision 1; same-class pairs: 1 + 3 = 4 ->
    # recall 2/4; unassigned 1 of 5
    _, precision, recall, unassigned = dg.score_partition(
        partition, ['a', 'a', 'b', 'b', 'b'], labelled
    )
    assert (precision, recall, unassigned) == (1.0, 0.5, pytest.approx(0.2))


def test_best_precision_at_takes_the_best_point_that_reaches_the_recall() -> None:
    curve = [
        (0.0, 0.5, 0.9, 0.0),
        (1.0, 0.8, 0.4, 0.0),
        (2.0, 0.95, 0.1, 0.0),
        (3.0, None, None, 0.0),
    ]
    assert dg.best_precision_at(curve, 0.3) == 0.8
    assert dg.best_precision_at(curve, 0.5) == 0.5
    assert dg.best_precision_at(curve, 0.05) == 0.95
    assert dg.best_precision_at(curve, 0.95) is None


def _two_class_vectors() -> tuple[np.ndarray, list[str]]:
    angles = [0, 4, 8, 90, 94, 98]
    return _unit_at(*angles), ['a', 'a', 'a', 'b', 'b', 'b']


@pytest.mark.parametrize('method', ['average', 'complete'])
def test_ahc_curve_reaches_perfect_precision_and_recall_on_separated_classes(method: str) -> None:
    x, truth = _two_class_vectors()
    curve = dg.ahc_curve(x @ x.T, truth, np.arange(6), method, n_points=10)
    # the cuts run from the median to the 99.5th percentile of merge heights; the last merge
    # (the two classes, ~90 degrees apart) is above all of them, so each cut leaves <= 2 groups
    assert dg.best_precision_at(curve, 1.0) == 1.0


def test_mutual_knn_graph_keeps_only_reciprocal_edges() -> None:
    # k = 1: 0-1 and 2-3 are mutual. k = 2 lists 0:[1,3] 1:[0,3] 2:[3,1] 3:[2,1]:
    # mutual pairs are (0,1), (1,3), (2,3); (0,3) and (1,2) are one-sided.
    assert dg.mutual_knn_graph(SIM4, 1) == ([(0, 1), (2, 3)], pytest.approx([0.9, 0.8]))
    edges, weights = dg.mutual_knn_graph(SIM4, 2)
    assert edges == [(0, 1), (1, 3), (2, 3)]
    assert weights == pytest.approx([0.9, 0.4, 0.8])


def _two_cliques(per_class: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    centres = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    x = np.vstack([c + 0.05 * rng.standard_normal((per_class, 4)) for c in centres])
    return dg.l2_normalize(x), np.repeat([0, 1], per_class)


def test_leiden_cpm_separates_two_cliques_and_drops_a_lone_vector() -> None:
    pytest.importorskip('leidenalg')
    x, _ = _two_cliques()
    x = np.vstack([x, dg.l2_normalize(np.array([[0, 0, 1.0, 1.0]]))])  # a vector near nobody
    part = dg.leiden_cpm_partition(x @ x.T, k=7, fraction=0.3)  # k = clique size - 1
    assert part[-1] == UNASSIGNED  # a cluster of one is under min_size
    assert len(set(part[:8])) == 1
    assert len(set(part[8:16])) == 1
    assert part[0] != part[8]
    assert (part[:16] >= 0).all()


def test_ecg_separates_two_cliques_and_is_repeatable() -> None:
    pytest.importorskip('igraph')
    x, _ = _two_cliques()
    first = pt.ecg_partition(x @ x.T, k=5, n_runs=8)
    assert len(set(first[:8])) == 1
    assert len(set(first[8:])) == 1
    assert first[0] != first[8]
    assert np.array_equal(first, pt.ecg_partition(x @ x.T, k=5, n_runs=8))


def test_ecg_is_repeatable_on_overlapping_blobs() -> None:
    # Louvain breaks ties with igraph's random number generator; unseeded, two runs on blobs
    # that overlap differ. The seed argument must fix them.
    pytest.importorskip('igraph')
    rng = np.random.default_rng(2)
    x = dg.l2_normalize(rng.standard_normal((120, 6)) + np.repeat(np.eye(6)[:3], 40, axis=0) * 1.2)
    runs = [pt.ecg_partition(x @ x.T, k=10, n_runs=6, seed=4) for _ in range(4)]
    assert all(np.array_equal(runs[0], r) for r in runs[1:])


# ---------------------------------------------------------------- partitions helpers


def test_drop_small_unassigns_clusters_under_the_minimum() -> None:
    got = pt.drop_small(np.array([0, 0, 0, 1, 1, 2, UNASSIGNED]), min_size=3)
    assert got.tolist() == [0, 0, 0, UNASSIGNED, UNASSIGNED, UNASSIGNED, UNASSIGNED]


def test_cut_groups_never_joins_two_groups_and_uses_one_global_cut() -> None:
    # Two identical-direction points in different groups would merge globally; per group they
    # cannot. Group "p" holds rows 0, 1 (close) and group "q" rows 2, 3 (close to 0 and 1).
    x = _unit_at(0, 3, 1, 4)
    sim = x @ x.T
    groups = ['p', 'p', 'q', 'q']
    per_group = pt.group_linkages(groups, sim)
    labels = pt.cut_groups(len(groups), per_group, cut=0.01)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    tight = pt.cut_groups(len(groups), per_group, cut=0.0001)  # below the pair distance: all apart
    assert len(set(tight.tolist())) == 4


def test_ahc_cuts_are_monotone_in_cluster_count() -> None:
    x, _ = _two_class_vectors()
    counts = [len(set(p.tolist())) for _, p in pt.ahc_cuts(x @ x.T, 'average', n_cuts=8)]
    assert counts == sorted(counts, reverse=True)
