"""Face-grouping scoring harness (scripts/research/face_grouping): every expected value is worked out beside it."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.face_grouping import (
    face_table as ft,
    grouping as gp,
    incremental as inc,
    quality as q,
    scores as sc,
    verification as ver,
)


def unit(*degrees: float) -> np.ndarray:
    a = np.radians(np.array(degrees))
    return np.stack([np.cos(a), np.sin(a)], axis=1).astype(np.float32)


# ------------------------------------------------------------------ face table and pairs


def test_main_face_takes_the_detection_nearest_the_centre_and_reports_zero():
    face_img = np.array([0, 0, 2])
    box = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.7, 0.7], [0.4, 0.4, 0.6, 0.6]])
    chosen, n_faces = ft.main_face(face_img, box, 3)
    assert chosen.tolist() == [1, -1, 2]
    assert n_faces.tolist() == [2, 0, 1]


def test_lfw_pairs_are_expanded_to_file_names_and_ten_contiguous_folds():
    text = '1\t2\nAbel\t1\t4\nAbel\t2\tBob\t3\n'
    pairs = ft.parse_lfw_pairs(text)
    assert pairs == [
        ('Abel_0001.jpg', 'Abel_0004.jpg', 1),
        ('Abel_0002.jpg', 'Bob_0003.jpg', 0),
    ]


def test_cross_pairs_take_two_lines_per_pair_and_a_nonzero_column_means_same():
    text = 'A_1.jpg 7\nA_2.jpg 7\nA_3.jpg 0\nB_1.jpg 0\n'
    assert ft.parse_cross_pairs(text) == [('A_1.jpg', 'A_2.jpg', 1), ('A_3.jpg', 'B_1.jpg', 0)]


def test_folds_rank_each_class_in_blocks_so_every_fold_is_balanced():
    labels = np.array(
        [1] * 20 + [0] * 20
    )  # positives block then negatives block, as CALFW and CPLFW ship
    folds = ft.fold_ids(labels, folds=10)
    assert folds[:4].tolist() == [0, 0, 1, 1]
    assert folds[20:24].tolist() == [0, 0, 1, 1]
    assert np.bincount(folds).tolist() == [4] * 10


# ------------------------------------------------------------------ verification


def test_fold_accuracy_is_one_for_separable_scores_and_tar_counts_accepted_positives():
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.0] * 2)
    labels = np.array([1, 1, 1, 0, 0, 0] * 2)
    folds = np.array([0] * 6 + [1] * 6)
    acc = ver.fold_accuracy(scores, labels, folds)
    assert acc['accuracy'] == 1.0
    # far=1/3 over three train negatives (0.1, 0.2, 0.0): threshold just above 0.2 accepts 1 negative
    # of 3; every positive (0.7 and up) passes
    tar = ver.tar_at_far(scores, labels, folds, 1 / 3)
    assert tar == 1.0


def test_tar_is_below_one_when_positives_sit_under_the_negative_quantile():
    scores = np.array([0.9, 0.3, 0.5, 0.4, 0.6, 0.35] * 2)
    labels = np.array([1, 1, 0, 0, 0, 0] * 2)
    folds = np.array([0] * 6 + [1] * 6)
    # train negatives 0.5, 0.4, 0.6, 0.35: FAR 0 needs threshold above 0.6 -> only 0.9 accepted
    assert ver.tar_at_far(scores, labels, folds, 0.0) == pytest.approx(0.5)


# ------------------------------------------------------------------ identity scores


def test_identity_report_counts_wrong_merges_splits_singletons():
    truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    cluster = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    r = sc.identity_report(truth, cluster, min_images=4)
    assert r['clusters'] == 2
    assert r['identities'] == 2
    assert r['wrong_merge_rate'] == 1.0  # both clusters hold two faces of each identity
    assert r['split_rate'] == 1.0  # both identities sit in two clusters
    assert r['singleton_share'] == 0.0
    assert r['bcubed_precision'] == pytest.approx(0.5)
    assert r['pair_precision'] == pytest.approx(4 / 12)


def test_singleton_share_and_task_recall_need_purity():
    truth = np.array([0] * 5 + [1] * 5)
    cluster = np.array([0, 0, 0, 0, 1, 2, 2, 2, 2, 1])
    r = sc.identity_report(truth, cluster, min_images=5)
    assert r['task_recall'] == pytest.approx(0.8)  # 4 of 5 in a pure cluster, for both identities
    assert r['singleton_share'] == 0.0
    impure = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0])
    assert sc.identity_report(truth, impure, min_images=5)['task_recall'] == pytest.approx(0.4)
    r2 = sc.identity_report(np.array([0, 0, 1, 1]), np.array([0, 1, 2, 2]), min_images=2)
    assert r2['singleton_share'] == pytest.approx(0.5)


def test_unassigned_faces_count_as_singletons_in_the_report():
    r = sc.identity_report(np.array([0, 0, 1]), np.array([5, 5, -1]), min_images=2)
    assert r['singleton_share'] == pytest.approx(1 / 3)


def test_best_point_picks_highest_recall_at_the_precision_floor():
    rows = [
        {'bcubed_precision': 0.99, 'bcubed_recall': 0.2},
        {'bcubed_precision': 0.96, 'bcubed_recall': 0.6},
        {'bcubed_precision': 0.90, 'bcubed_recall': 0.9},
    ]
    assert sc.best_point(rows, 0.95) is rows[1]
    assert sc.best_point(rows, 0.99) is rows[0]
    assert sc.best_point(rows, 0.999) is None


# ------------------------------------------------------------------ methods


def graph(x: np.ndarray, k: int = 5, floor: float = 0.3) -> gp.Graph:
    return gp.build_graph(x, k=k, floor=floor)


def same(labels: np.ndarray, *idx: int) -> bool:
    return len({int(labels[i]) for i in idx}) == 1


def test_single_linkage_chains_but_average_and_complete_do_not():
    x = unit(0, 10, 25, 90)
    single = gp.single_linkage(graph(x), 0.95)
    assert same(single, 0, 1, 2)
    assert not same(single, 0, 3)
    for method in ('average', 'complete'):
        lab = gp.agglomerative(x, method, 0.95)
        assert same(lab, 0, 1)
        assert not same(lab, 0, 2)
        assert not same(lab, 2, 3)


def test_immich_core_needs_min_faces_hits_and_leaves_sparse_faces_unassigned():
    x = unit(0, 1, 2, 3, 90, 91, 180)
    lab = gp.immich_core(graph(x), max_distance=0.1, min_faces=3)
    assert same(lab, 0, 1, 2, 3)
    assert lab[4] == gp.UNASSIGNED
    assert lab[5] == gp.UNASSIGNED
    assert lab[6] == gp.UNASSIGNED
    pair = gp.immich_core(graph(x), max_distance=0.1, min_faces=2)
    assert same(pair, 4, 5)
    assert not same(pair, 0, 4)
    assert pair[6] == gp.UNASSIGNED


def test_mutual_knn_needs_mutual_neighbours_and_merge_pass_joins_close_clusters():
    x = unit(0, 5, 12, 90)
    lab = gp.mutual_knn(graph(x), k=1, floor=0.5, merge_sim=None)
    assert same(lab, 0, 1)
    assert not same(lab, 1, 2)
    assert not same(lab, 0, 3)
    merged = gp.mutual_knn(graph(x), k=1, floor=0.5, merge_sim=0.98)
    assert same(merged, 0, 1, 2)
    assert not same(merged, 0, 3)


def test_chinese_whispers_splits_two_cliques_joined_below_the_edge_floor():
    pytest.importorskip('chinese_whispers')
    x = unit(0, 1, 2, 60, 61, 62)
    lab = gp.chinese_whisper_groups(graph(x), floor=0.9, seed=0)
    assert same(lab, 0, 1, 2)
    assert same(lab, 3, 4, 5)
    assert not same(lab, 0, 3)


def test_finch_levels_include_the_two_group_partition():
    pytest.importorskip('finch')
    x = unit(0, 1, 2, 3, 80, 81, 82, 83)
    levels = gp.finch_levels(x)
    assert any(
        len(set(lab.tolist())) == 2 and same(lab, 0, 1, 2, 3) and not same(lab, 0, 4)
        for lab in levels
    )


# ------------------------------------------------------------------ quality signals


def test_pose_proxy_is_nose_offset_over_eye_distance():
    lm = np.array(
        [
            [0.3, 0.4, 0.7, 0.4, 0.5, 0.6, 0.35, 0.8, 0.65, 0.8],
            [0.3, 0.4, 0.7, 0.4, 0.6, 0.6, 0.35, 0.8, 0.65, 0.8],
        ]
    )
    assert q.pose_offset(lm).tolist() == pytest.approx([0.0, 0.25])


def test_error_reject_area_is_lowest_when_errors_sit_at_low_quality():
    quality = np.array([1.0, 2.0, 3.0, 4.0])
    err = np.array([1, 1, 0, 0])
    good = q.error_reject_area(quality, err)
    assert good == pytest.approx((0.5 + 1 / 3) / 4)
    assert q.error_reject_area(-quality, err) > good  # reversed ordering rejects the wrong end


def test_quality_gate_lets_weak_faces_join_but_not_bridge():
    x = unit(0, 2, 21, 40, 42)
    quality = np.array([1, 1, 0.1, 1, 1.0])
    plain = gp.single_linkage(graph(x), 0.93)
    assert same(plain, 0, 4)  # the weak middle face chains the two groups
    gated = q.gated(x, quality, 0.5, lambda sub: gp.single_linkage(graph(sub), 0.93), join_sim=0.93)
    assert same(gated, 0, 1)
    assert same(gated, 3, 4)
    assert not same(gated, 0, 3)
    assert same(gated, 2, 0) or same(gated, 2, 3)


# ------------------------------------------------------------------ incremental


def test_incremental_assigns_to_nearest_centroid_then_clusters_the_leftovers():
    x = unit(0, 1, 2, 90, 91, 92)
    batches = [np.array([0, 1, 3]), np.array([2, 4, 5])]
    lab = inc.incremental(x, batches, lambda sub: gp.single_linkage(graph(sub), 0.9), join_sim=0.9)
    assert same(lab, 0, 1, 2)
    assert same(lab, 3, 4, 5)
    assert not same(lab, 0, 3)


# ------------------------------------------------------------------ driver plumbing (random vectors, no quality claim)


def _fake_set(root, name, fmt, rng):
    ids = ['Ann', 'Bob', 'Cy', 'Di', 'Ed', 'Flo']
    centres = {i: rng.normal(size=512) for i in ids}
    files, ident, emb = [], [], []
    for i in ids:
        for k in range(1, 13):
            files.append(f'{i}_{fmt(k)}.jpg')
            ident.append(i)
            emb.append(centres[i] + 0.8 * rng.normal(size=512))
    n = len(files)
    d = root / name
    d.mkdir()
    (d / 'manifest.json').write_text('{}')
    vec = np.array(emb, dtype=np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    np.savez(
        root / f'{name}.npz',
        files=np.array(files),
        identity=np.array(ident),
        width=np.full(n, 250),
        height=np.full(n, 250),
        face_img=np.arange(n),
        box=np.tile([0.3, 0.3, 0.7, 0.7], (n, 1)).astype(np.float32),
        landmarks=np.tile([0.4, 0.45, 0.6, 0.45, 0.5, 0.55, 0.42, 0.7, 0.58, 0.7], (n, 1)).astype(
            np.float32
        ),
        score=rng.uniform(0.5, 1, n).astype(np.float32),
        quality=rng.uniform(0, 1, n).astype(np.float32),
        emb=vec,
    )
    lines = []
    for p in range(20):
        a, b = ids[p % 6], ids[(p + 1) % 6]
        same = p % 2 == 0
        pair = (
            (f'{a}_{fmt(1)}.jpg', f'{a}_{fmt(2)}.jpg')
            if same
            else (f'{a}_{fmt(1)}.jpg', f'{b}_{fmt(1)}.jpg')
        )
        if name == 'lfw':
            lines.append(f'{a}\t1\t2' if same else f'{a}\t1\t{b}\t1')
        else:
            lines += [f'{pair[0]} {int(same)}', f'{pair[1]} {int(same)}']
    from scripts.research.face_grouping import datasets as ds

    (d / ds.PAIRS[name]).write_text('\n'.join(lines) + '\n')


def test_run_eval_produces_every_section_end_to_end(tmp_path):
    pytest.importorskip('finch')
    pytest.importorskip('chinese_whispers')
    from scripts.research.face_grouping import run_eval

    rng = np.random.default_rng(0)
    _fake_set(tmp_path, 'lfw', lambda k: f'{k:04d}', rng)
    _fake_set(tmp_path, 'calfw', lambda k: f'{k:04d}', rng)
    _fake_set(tmp_path, 'cplfw', lambda k: str(k), rng)
    out = tmp_path / 'out'
    import sys

    argv, sys.argv = sys.argv, ['run_eval', str(tmp_path), str(out)]
    try:
        assert run_eval.main() == 0
    finally:
        sys.argv = argv
    md = (out / 'results.md').read_text()
    for section in (
        '## A.',
        '## B/C. bed1_lfw',
        '## B/C. bed3_ge10_plus_singletons',
        '## D.',
        '## E.',
    ):
        assert section in md


def test_run_spec_on_a_lone_face_returns_one_cluster_for_every_method():
    from scripts.research.face_grouping import sweep as sw

    one = unit(0)
    for spec in (
        {'method': 'average', 'params': {'sim': 0.5}},
        {'method': 'single', 'params': {'sim': 0.5}},
        {'method': 'immich', 'params': {'max_distance': 0.3, 'min_faces': 3}},
    ):
        assert sw.run_spec(spec, one).shape == (1,)
