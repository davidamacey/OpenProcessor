"""Parity metrics on synthetic boxes, vectors, labellings and strings (pure logic, no stack)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.bench import parity_metrics as pm


def _unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------- boxes


def test_iou_known_values() -> None:
    a = np.array([[0.0, 0.0, 0.5, 0.5]], dtype=np.float32)
    b = np.array([[0.0, 0.0, 0.5, 0.5], [0.25, 0.0, 0.75, 0.5], [0.6, 0.6, 0.9, 0.9]], np.float32)
    iou = pm.box_iou(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(1 / 3)
    assert iou[0, 2] == 0.0


def test_match_boxes_is_greedy_within_class() -> None:
    ref = np.array([[0.0, 0.0, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]], np.float32)
    cand = np.array([[0.01, 0.0, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]], np.float32)
    pairs, miss_ref, miss_cand = pm.match_boxes(ref, ['a', 'b'], cand, ['a', 'b'])
    assert sorted(pairs) == [(0, 0), (1, 1)]
    assert not miss_ref
    assert not miss_cand


def test_match_boxes_never_crosses_classes() -> None:
    box = np.array([[0.1, 0.1, 0.6, 0.6]], np.float32)
    pairs, miss_ref, miss_cand = pm.match_boxes(box, ['cat'], box, ['dog'])
    assert pairs == []
    assert miss_ref == [0]
    assert miss_cand == [0]


def test_match_boxes_takes_the_best_overlap_first() -> None:
    ref = np.array([[0.0, 0.0, 0.5, 0.5]], np.float32)
    cand = np.array([[0.0, 0.0, 0.4, 0.5], [0.0, 0.0, 0.5, 0.5]], np.float32)
    pairs, _, miss_cand = pm.match_boxes(ref, ['a'], cand, ['a', 'a'])
    assert pairs == [(0, 1)]
    assert miss_cand == [0]


def _items(rows: list[tuple[str, str, float, float, float, float, float]]) -> dict:
    return {
        'image': np.array([r[0] for r in rows]),
        'cls': np.array([r[1] for r in rows]),
        'box': np.array([r[3:7] for r in rows], np.float32).reshape(-1, 4),
        'score': np.array([r[2] for r in rows], np.float32),
    }


def _box_metrics(ref: dict, cand: dict) -> dict:
    return pm.box_metrics(
        (ref['image'], ref['box'], ref['cls'], ref['score']),
        (cand['image'], cand['box'], cand['cls'], cand['score']),
    )


def test_box_metrics_identical_sets() -> None:
    ref = _items([('a', 'car', 0.9, 0.1, 0.1, 0.5, 0.5), ('b', 'dog', 0.8, 0.2, 0.2, 0.9, 0.9)])
    out = _box_metrics(ref, ref)
    assert out['images_identical_share'] == 1.0
    assert out['unmatched_confident'] == 0
    assert out['item_count_class_identical_share'] == 1.0


def test_box_metrics_shifted_box_breaks_identity_but_still_matches() -> None:
    ref = _items([('a', 'car', 0.9, 0.0, 0.0, 0.5, 0.5), ('b', 'dog', 0.8, 0.2, 0.2, 0.9, 0.9)])
    cand = _items([('a', 'car', 0.9, 0.05, 0.0, 0.55, 0.5), ('b', 'dog', 0.8, 0.2, 0.2, 0.9, 0.9)])
    out = _box_metrics(ref, cand)
    assert out['images_identical_share'] == 0.5
    assert out['unmatched_confident'] == 0
    assert out['iou_min'] < 0.95


def test_box_metrics_score_delta_breaks_identity() -> None:
    ref = _items([('a', 'car', 0.90, 0.0, 0.0, 0.5, 0.5)])
    cand = _items([('a', 'car', 0.95, 0.0, 0.0, 0.5, 0.5)])
    assert _box_metrics(ref, cand)['images_identical_share'] == 0.0


def test_box_metrics_counts_confident_unmatched_only() -> None:
    ref = _items([('a', 'car', 0.9, 0.0, 0.0, 0.5, 0.5), ('a', 'dog', 0.3, 0.6, 0.6, 0.9, 0.9)])
    cand = _items([('a', 'car', 0.9, 0.0, 0.0, 0.5, 0.5)])
    out = _box_metrics(ref, cand)
    assert out['unmatched_confident'] == 0  # the lost box scored 0.3
    cand_extra = _items(
        [('a', 'car', 0.9, 0.0, 0.0, 0.5, 0.5), ('a', 'cat', 0.8, 0.6, 0.6, 0.9, 0.9)]
    )
    assert _box_metrics(ref, cand_extra)['unmatched_confident'] == 1


def test_box_metrics_class_change_is_a_class_mismatch() -> None:
    ref = _items([('a', 'car', 0.9, 0.0, 0.0, 0.5, 0.5)])
    cand = _items([('a', 'truck', 0.9, 0.0, 0.0, 0.5, 0.5)])
    out = _box_metrics(ref, cand)
    assert out['item_count_class_identical_share'] == 0.0
    assert out['unmatched_confident'] == 2


def test_pair_images_pairs_are_global_indices() -> None:
    ref = _items([('a', 'car', 0.9, 0.0, 0.0, 0.5, 0.5), ('b', 'dog', 0.8, 0.2, 0.2, 0.9, 0.9)])
    cand = _items([('b', 'dog', 0.8, 0.2, 0.2, 0.9, 0.9), ('a', 'car', 0.9, 0.0, 0.0, 0.5, 0.5)])
    pairs = pm.pair_items(
        (ref['image'], ref['box'], ref['cls']), (cand['image'], cand['box'], cand['cls'])
    )
    assert pairs.tolist() == [[0, 1], [1, 0]]


# ---------------------------------------------------------------- vectors


def test_cosine_summary_of_noisy_copy() -> None:
    rng = np.random.default_rng(0)
    a = _unit(rng, 500, 32)
    b = a + rng.normal(scale=0.01, size=a.shape).astype(np.float32)
    s = pm.cosine_summary(pm.cosine_rows(a, b))
    assert s['n'] == 500
    assert s['min'] <= s['p01'] <= s['median'] <= 1.0
    assert s['median'] > 0.99
    assert pm.cosine_summary(pm.cosine_rows(a, a))['min'] == pytest.approx(1.0, abs=1e-6)


def test_cosine_summary_detects_a_swapped_row() -> None:
    rng = np.random.default_rng(1)
    a = _unit(rng, 100, 32)
    b = a.copy()
    b[7] = -a[7]
    assert pm.cosine_summary(pm.cosine_rows(a, b))['min'] == pytest.approx(-1.0, abs=1e-5)


def test_knn_agreement_identical_and_noisy() -> None:
    rng = np.random.default_rng(2)
    a = _unit(rng, 300, 24)
    same = pm.knn_agreement(a, a, k=10, n_queries=100, seed=3, dup_threshold=0.98)
    assert same['overlap_at_k'] == 1.0
    assert same['top1_recall_at_k'] == 1.0
    assert same['near_dup_agreement'] == 1.0
    b = _unit(rng, 300, 24)  # unrelated space
    other = pm.knn_agreement(a, b, k=10, n_queries=100, seed=3, dup_threshold=0.98)
    assert other['overlap_at_k'] < 0.2


def test_knn_agreement_excludes_the_query_itself() -> None:
    a = np.eye(4, dtype=np.float32)
    got = pm.knn_indices(a, np.array([0]), k=3)
    assert 0 not in got[0]


def test_near_dup_agreement_sees_a_threshold_crossing() -> None:
    base = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0]], np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    moved = base.copy()
    moved[1] = np.array([0.8, 0.6], np.float32)  # no longer a near duplicate of row 0
    out = pm.knn_agreement(base, moved, k=2, n_queries=3, seed=0, dup_threshold=0.98)
    assert out['near_dup_agreement'] < 1.0


# ---------------------------------------------------------------- clusters


def test_ari_is_label_permutation_invariant() -> None:
    a = np.array([0, 0, 1, 1, 2, 2, 2])
    perm = np.array([5, 5, 3, 3, 9, 9, 9])
    assert pm.adjusted_rand_index(a, perm) == pytest.approx(1.0)
    assert pm.normalized_mutual_info(a, perm) == pytest.approx(1.0)


def test_ari_penalises_a_split_cluster() -> None:
    a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    b = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    assert pm.adjusted_rand_index(a, b) < 0.5
    assert 0.0 < pm.normalized_mutual_info(a, b) < 1.0


def test_ari_known_value() -> None:
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 0, 1, 2])
    assert pm.adjusted_rand_index(a, b) == pytest.approx(0.5714285714)


def test_purity_weighted_by_cluster_size() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1])
    truth = np.array(['a', 'a', 'a', 'b', 'c', 'd'])
    assert pm.purity(labels, truth) == pytest.approx((3 + 1) / 6)


def test_assignment_agreement_is_one_for_identical_vectors() -> None:
    rng = np.random.default_rng(4)
    centres = _unit(rng, 4, 16)
    labels = rng.integers(0, 4, size=200)
    vec = centres[labels] + rng.normal(scale=0.02, size=(200, 16)).astype(np.float32)
    assert pm.assignment_agreement(vec, vec, labels) == 1.0
    scrambled = _unit(rng, 200, 16)
    assert pm.assignment_agreement(vec, scrambled, labels) < 0.6


def test_kmeans_objective_is_lower_for_tight_clusters() -> None:
    rng = np.random.default_rng(5)
    centres = _unit(rng, 3, 16)
    labels = rng.integers(0, 3, size=150)
    tight = centres[labels] + rng.normal(scale=0.01, size=(150, 16)).astype(np.float32)
    loose = centres[labels] + rng.normal(scale=0.3, size=(150, 16)).astype(np.float32)
    assert pm.kmeans_objective(tight, labels) < pm.kmeans_objective(loose, labels)


# ---------------------------------------------------------------- strings and labels


def test_agreement_share_counts_equal_strings() -> None:
    assert pm.agreement_share(['a', 'b', 'c'], ['a', 'x', 'c']) == pytest.approx(2 / 3)
    assert pm.agreement_share([], []) == 1.0


def test_text_agreement_matches_boxes_then_compares_exactly() -> None:
    ref = (
        np.array(['a', 'a']),
        np.array([[0.0, 0.0, 0.5, 0.2], [0.0, 0.5, 0.5, 0.7]], np.float32),
        np.array(['STOP', 'EXIT']),
    )
    cand = (
        np.array(['a', 'a']),
        np.array([[0.0, 0.0, 0.5, 0.2], [0.0, 0.5, 0.5, 0.7]], np.float32),
        np.array(['STOP', 'EXIT.']),
    )
    out = pm.text_agreement(ref, cand)
    assert out['matched'] == 2
    assert out['exact_share'] == 0.5
    assert out['char_error_rate'] > 0.0


def test_cosine_ignores_vector_length() -> None:
    a = np.array([[2.0, 0.0], [3.0, 4.0]])
    b = np.array([[0.5, 0.0], [4.0, 3.0]])
    assert pm.cosine_rows(a, b).tolist() == pytest.approx([1.0, 0.96])


def test_knn_label_vote_and_precision_follow_the_vectors() -> None:
    rng = np.random.default_rng(6)
    centres = _unit(rng, 3, 16)
    labels = np.repeat(np.array(['a', 'b', 'c']), 60)
    vec = centres[np.repeat(np.arange(3), 60)] + rng.normal(scale=0.02, size=(180, 16))
    vec = vec.astype(np.float32)
    same = pm.knn_agreement(
        vec,
        vec,
        k=10,
        n_queries=100,
        seed=1,
        dup_threshold=0.98,
        labels=labels,
        truth=(labels, labels),
    )
    assert same['label_vote_agreement'] == 1.0
    assert same['precision_ref'] == 1.0
    assert same['precision_drop_points'] == 0.0
    scrambled = _unit(rng, 180, 16)
    worse = pm.knn_agreement(
        vec,
        scrambled,
        k=10,
        n_queries=100,
        seed=1,
        dup_threshold=0.98,
        labels=labels,
        truth=(labels, labels),
    )
    assert worse['label_vote_agreement'] < 0.6
    assert worse['precision_drop_points'] > 50.0


def test_knn_precision_ignores_queries_without_ground_truth() -> None:
    rng = np.random.default_rng(7)
    vec = _unit(rng, 50, 8)
    none = np.array(['none'] * 50)
    out = pm.knn_agreement(
        vec, vec, k=5, n_queries=20, seed=0, dup_threshold=0.98, truth=(none, none)
    )
    assert 'precision_drop_points' not in out
