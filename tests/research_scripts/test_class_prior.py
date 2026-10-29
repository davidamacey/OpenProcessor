"""Hand-computed cases for the class-partitioned clustering arm and its scoring.

Every expected value is worked out on paper in the comment beside it.
"""

from __future__ import annotations

import json
import math
from argparse import Namespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from scripts.research.cluster_diag import class_prior as cp, class_prior_bars as bars
from scripts.research.cluster_diag.dataset import detector_keys, load_bundle


if TYPE_CHECKING:
    from pathlib import Path


def _unit_at(*degrees: float) -> np.ndarray:
    rad = np.radians(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


# ---------------------------------------------------------------- partitions


def test_within_key_clusters_never_cross_a_key_even_when_a_global_cut_would_merge() -> None:
    # key a: 0, 10, 90 degrees; key b: 5, 95 degrees. 1 - cos(10 deg) = 0.0152 < 0.05, so
    # a0 and a10 merge; a90 is 1 - cos(90) = 1 away from both. b5 and b95 are 1 apart.
    # b5 is only 1 - cos(5) = 0.0038 from a0, which a global cut at 0.05 would merge.
    vectors = _unit_at(0, 10, 90, 5, 95)
    part = cp.WithinKeyClusterer(vectors, ['a', 'a', 'a', 'b', 'b']).partition(0.05)
    assert part[0] == part[1]
    assert len(set(part.tolist())) == 4
    assert part[3] != part[0]
    assert part[3] != part[4]


def test_a_higher_cut_merges_inside_the_key_only() -> None:
    vectors = _unit_at(0, 10, 90, 5, 95)
    keys = ['a', 'a', 'a', 'b', 'b']
    # at cut 1.2 every pair inside a key is under the cut (largest is 1 - cos(90) = 1), so
    # two clusters, one per key
    part = cp.WithinKeyClusterer(vectors, keys).partition(1.2)
    assert len(set(part.tolist())) == 2
    assert part[0] == part[1] == part[2]
    assert part[3] == part[4]
    assert part[0] != part[3]


def test_key_only_partition_is_one_cluster_per_key() -> None:
    assert cp.key_only_partition(['b', 'a', 'b']).tolist() == [1, 0, 1]


def test_trivial_partitions_cover_the_three_baselines() -> None:
    got = cp.trivial_partitions(['x', 'y', 'x'])
    assert got['detector_class_only'].tolist() == [0, 1, 0]
    assert got['all_singletons'].tolist() == [0, 1, 2]
    assert got['one_cluster'].tolist() == [0, 0, 0]


def test_cut_grids() -> None:
    assert cp.uniform_cut_grid(0.1, 0.3, 0.1) == pytest.approx([0.1, 0.2, 0.3])
    grid = cp.quantile_cut_grid(_unit_at(*range(0, 180, 3)))
    assert len(grid) == cp.QUANTILE_GRID_POINTS
    assert grid == sorted(grid)


# ---------------------------------------------------------------- scoring views


def test_score_block_matches_the_hand_counted_pairs_and_bcubed() -> None:
    truth = np.array(['a', 'a', 'b', 'b'])
    part = np.array([0, 0, 0, 1])
    got = cp.score_block(truth, part, np.ones(4, dtype=bool))
    # cluster 0 = {a, a, b}: 3 pairs, 1 of them a-a -> pair precision 1/3; same-class pairs
    # a-a and b-b = 2, one kept together -> pair recall 1/2
    assert got['pair_precision'] == pytest.approx(1 / 3)
    assert got['pair_recall'] == pytest.approx(1 / 2)
    # BCubed precision: a, a are 2/3 pure; b in cluster 0 is 1/3; the lone b is 1 -> 8/3 / 4
    assert got['bcubed_precision'] == pytest.approx(2 / 3)
    # BCubed recall: a, a find both a's (1 each); each b finds 1 of 2 -> (1 + 1 + .5 + .5) / 4
    assert got['bcubed_recall'] == pytest.approx(3 / 4)
    assert got['bcubed_f'] == pytest.approx(2 * (2 / 3) * (3 / 4) / (2 / 3 + 3 / 4))


def test_person_pair_share_counts_same_label_pairs() -> None:
    truth = np.array(['person'] * 3 + ['car'] * 2)
    # person pairs 3, car pairs 1
    assert cp.person_pair_share(truth, np.ones(5, dtype=bool)) == pytest.approx(3 / 4)
    # without one person: person pairs 1, car pairs 1
    mask = np.array([True, True, False, True, True])
    assert cp.person_pair_share(truth, mask) == pytest.approx(1 / 2)


def test_views_split_person_from_the_rest_and_report_each_stratum() -> None:
    truth = np.array(['person', 'person', 'car', 'car'])
    part = np.array([0, 0, 1, 2])  # the two cars are split
    strata = {'low_res': np.array([False, False, True, True])}
    got = cp.views(truth, part, np.ones(4, dtype=bool), strata)
    assert got['person']['pair_recall'] == pytest.approx(1.0)
    assert got['non_person']['n'] == 2
    assert got['non_person']['pair_recall'] == pytest.approx(0.0)
    assert got['non_person']['bcubed_recall'] == pytest.approx(0.5)  # each car finds 1 of 2
    assert got['stratum:low_res']['n'] == 2
    assert got['all']['ami'] is not None


# ---------------------------------------------------------------- product bars


def _views(**blocks: dict[str, float]) -> dict[str, dict[str, float | int | None]]:
    return {name.replace('__', ':'): dict(b) for name, b in blocks.items()}  # type: ignore[misc]


def test_product_bars_pass_and_fail_each_threshold_on_its_own() -> None:
    good = _views(
        all={'bcubed_precision': 0.95, 'bcubed_recall': 0.40, 'pair_recall': 0.10},
        non_person={'bcubed_precision': 0.90},
        stratum__low_res={'bcubed_precision': 0.90},
    )
    ivf = _views(stratum__low_res={'bcubed_precision': 0.80})
    rows = {
        r['metric']: r for r in bars.product_bars(good, ivf, permutation_ari=1.0, protected_moved=0)
    }
    assert all(r['pass'] for r in rows.values())
    worse = _views(
        all={'bcubed_precision': 0.89, 'bcubed_recall': 0.36, 'pair_recall': 0.06},
        non_person={'bcubed_precision': 0.87},
        stratum__low_res={'bcubed_precision': 0.70},
    )
    rows = {
        r['metric']: r
        for r in bars.product_bars(worse, ivf, permutation_ari=0.99, protected_moved=2)
    }
    assert not any(r['pass'] for r in rows.values())


def test_bar_edges_are_inclusive() -> None:
    edge = _views(
        all={'bcubed_precision': 0.90, 'bcubed_recall': 0.37, 'pair_recall': 3 * 0.022},
        non_person={'bcubed_precision': 0.88},
        stratum__low_res={'bcubed_precision': 0.85},
    )
    rows = {
        r['metric']: r
        for r in bars.product_bars(edge, edge, permutation_ari=1.0, protected_moved=0)
    }
    assert all(r['pass'] for r in rows.values())


def test_a_stratum_below_ivf_fails_even_when_the_others_pass() -> None:
    within = _views(
        all={'bcubed_precision': 0.95, 'bcubed_recall': 0.4, 'pair_recall': 0.1},
        non_person={'bcubed_precision': 0.9},
        stratum__low_res={'bcubed_precision': 0.9},
        stratum__rare={'bcubed_precision': 0.90},
    )
    ivf = _views(
        stratum__low_res={'bcubed_precision': 0.8}, stratum__rare={'bcubed_precision': 0.92}
    )
    assert bars.stratum_precision_gaps(within, ivf) == pytest.approx(
        {'stratum:low_res': 0.1, 'stratum:rare': -0.02}
    )
    rows = {
        r['metric']: r
        for r in bars.product_bars(within, ivf, permutation_ari=1.0, protected_moved=0)
    }
    row = rows['min_stratum_bcubed_precision_minus_ivf']
    assert not row['pass']
    assert row['worst_stratum'] == 'stratum:rare'


def test_equal_stratum_precision_passes_despite_float_noise() -> None:
    within = _views(stratum__rare={'bcubed_precision': 0.9166666666666666})
    ivf = _views(stratum__rare={'bcubed_precision': 0.9166666666666667})
    assert bars.stratum_precision_gaps(within, ivf) == {'stratum:rare': 0.0}


def test_a_missing_stratum_never_passes() -> None:
    within = _views(all={'bcubed_precision': 0.95, 'bcubed_recall': 0.4, 'pair_recall': 0.1})
    rows = {
        r['metric']: r
        for r in bars.product_bars(within, {}, permutation_ari=1.0, protected_moved=0)
    }
    assert rows['bcubed_precision_low_res']['value'] is None
    assert not rows['bcubed_precision_low_res']['pass']


def test_bar_shortfalls_and_the_reopen_margin() -> None:
    rows = [
        {'metric': 'bcubed_recall_all', 'bar': 0.37, 'value': 0.357, 'pass': False},
        {'metric': 'bcubed_precision_all', 'bar': 0.90, 'value': 0.85, 'pass': False},
        {'metric': 'protected_items_moved', 'bar': 0, 'value': 2, 'pass': False},
        {'metric': 'permutation_ari', 'bar': 1.0, 'value': 1.0, 'pass': True},
        {'metric': 'bcubed_precision_low_res', 'bar': 0.85, 'value': None, 'pass': False},
    ]
    got = bars.bar_shortfalls(rows)
    assert got['bcubed_recall_all'] == pytest.approx(0.013)
    assert got['bcubed_precision_all'] == pytest.approx(0.05)
    assert got['protected_items_moved'] == 2
    assert got['permutation_ari'] == pytest.approx(0.0)
    assert bars.bars_missed_by_more_than(rows) == [
        'bcubed_precision_all',
        'bcubed_precision_low_res',
        'protected_items_moved',
    ]


def test_permutation_ari_is_one_when_the_fit_is_a_function_of_the_sorted_pool() -> None:
    vectors = np.arange(40, dtype=np.float32).reshape(20, 2)
    ids = [f'c{i:02d}' for i in range(20)]
    keys = ['k'] * 20
    # the code of a row is its position in the sorted pool: right only if both fits are sorted

    def by_position(v: Any, k: Any) -> np.ndarray:
        return np.arange(len(v)) % 2

    assert bars.permutation_ari(by_position, vectors, keys, ids, 7) == pytest.approx(1.0)


def test_permutation_ari_drops_for_a_fit_that_depends_on_the_row_order() -> None:
    vectors = np.arange(40, dtype=np.float32).reshape(20, 2)
    ids = [f'c{i:02d}' for i in range(20)]
    calls: list[int] = []

    def drifting(v: Any, k: Any) -> np.ndarray:
        calls.append(1)
        return np.arange(len(v)) % (len(calls) + 1)  # a different split on every call

    assert bars.permutation_ari(drifting, vectors, ['k'] * 20, ids, 7) < 1.0


def test_fold_report_chooses_on_one_half_and_scores_the_other() -> None:
    truth = np.array(['a', 'a', 'b', 'b'] * 4)
    perfect = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    merged = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    first = np.array([True] * 8 + [False] * 8)
    report = bars.fold_report(
        {'f': [(0.2, perfect), (0.4, merged)]},
        {'ivf_refit': merged},
        truth,
        {},
        first,
        bar_family='f',
        permutation_ari_value=1.0,
        protected_moved=0,
        targets=(0.1,),
    )
    fold1, fold2 = report['fold1_calibration_to_heldout'], report['fold2_heldout_to_calibration']
    cell = fold1['families']['f']['0.1']
    # merged pairs a-b are false, so the perfect partition has the higher precision on the
    # fitting half and is the one chosen
    assert cell['cut'] == 0.2
    assert cell['views']['all']['n'] == 8
    assert fold1['fit_labelled'] == fold2['scored_labelled'] == 8
    assert {r['metric'] for r in cell['bars']} >= {'bcubed_precision_all', 'permutation_ari'}
    assert fold1['references']['ivf_refit']['all']['pair_precision'] == pytest.approx(
        1 / 3
    )  # merged clusters of {a,a,b,b}: 6 pairs, 2 a-a/b-b -> 1/3


# ---------------------------------------------------------------- bundle helpers


def test_detector_keys_use_the_product_name_rule_and_mark_a_missing_class() -> None:
    meta = [
        {'detector_class_name': 'Traffic Light'},
        {'detector_class_name': 'traffic_light'},
        {'detector_class_name': None},
        {},
    ]
    assert detector_keys(meta).tolist() == ['traffic_light', 'traffic_light', 'none', 'none']


def test_load_bundle_reads_the_protected_flag(tmp_path: Path) -> None:
    stem = tmp_path / 'p'
    meta = [
        {'crop_id': f'c{i}', 'image_id': 'i', 'image_path': '/x/a.jpg', 'bbox_norm': [0, 0, 1, 1]}
        for i in range(2)
    ]
    np.save(stem.with_suffix('.npy'), np.eye(2, dtype=np.float32))
    stem.with_suffix('.meta.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in meta))
    labels = tmp_path / 'l.jsonl'
    labels.write_text(
        json.dumps({'item_id': 'c0', 'label': 'x', 'stratum': 's', 'protected': True})
        + '\n'
        + json.dumps({'item_id': 'c1', 'label': 'x', 'stratum': 's'})
        + '\n'
    )
    gt = tmp_path / 'g.json'
    gt.write_text(json.dumps({'images': [{'file_name': 'a.jpg', 'width': 10, 'height': 10}]}))
    assert load_bundle(stem, labels, gt).protected.tolist() == [True, False]


# ---------------------------------------------------------------- the committed arms


def _project(tmp_path: Path, per_class: int = 60) -> tuple[Path, Path, Path, Path]:
    """Three separated classes of unique source images, detector class = label."""
    rng = np.random.default_rng(3)
    meta, vecs, labels, part = [], [], [], []
    for ci, name in enumerate(['car', 'bus', 'dog']):
        for j in range(per_class):
            idx = ci * per_class + j
            meta.append(
                {
                    'crop_id': f'c{idx:04d}',
                    'image_id': f'im{idx}',
                    'image_path': f'/x/{idx}.jpg',
                    'bbox_norm': [0, 0, 0.5, 0.5],
                    'detector_class_name': name,
                }
            )
            vecs.append(np.eye(16)[ci] + 0.08 * rng.standard_normal(16))
            lab = {'item_id': f'c{idx:04d}', 'label': name, 'stratum': 'rare'}
            labels.append(lab | {'group': f'im{idx}'})
            part.append({'item_id': f'c{idx:04d}', 'cluster_id': ci})
    stem = tmp_path / 'syn'
    np.save(stem.with_suffix('.npy'), np.asarray(vecs, dtype=np.float32))
    stem.with_suffix('.meta.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in meta))
    lab_path = tmp_path / 'labels.jsonl'
    lab_path.write_text(''.join(json.dumps(r) + '\n' for r in labels))
    gt = tmp_path / 'gt.json'
    gt.write_text(
        json.dumps(
            {
                'images': [
                    {'file_name': f'{i}.jpg', 'width': 64, 'height': 64}
                    for i in range(3 * per_class)
                ]
            }
        )
    )
    stored = tmp_path / 'stored.jsonl'
    stored.write_text(''.join(json.dumps(r) + '\n' for r in part))
    return stem, lab_path, gt, stored


def test_export_partition_arms_cut_inside_the_detector_class(tmp_path: Path) -> None:
    from scripts.research.cluster_diag import export_partition

    stem, _, _, _ = _project(tmp_path, per_class=10)
    out = tmp_path / 'p.jsonl'
    base = ['--bundle', str(stem), '--out', str(out)]
    assert export_partition.main([*base, '--arm', 'detector_class']) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    # the detector class alone: three clusters of ten, one per class, whatever the vectors say
    assert [len({r['cluster_id'] for r in rows[c * 10 : (c + 1) * 10]}) for c in range(3)] == [
        1,
        1,
        1,
    ]
    assert len({r['cluster_id'] for r in rows}) == 3
    assert (
        export_partition.main([*base, '--arm', 'ahc_average_within_class', '--param', '1.9']) == 0
    )
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len({r['cluster_id'] for r in rows}) == 3  # cut 1.9 merges all of a key
    assert (
        export_partition.main([*base, '--arm', 'ahc_average_within_class', '--param', '0.0001'])
        == 0
    )
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len({r['cluster_id'] for r in rows}) == 30  # a tiny cut leaves every crop alone
    with pytest.raises(SystemExit):
        export_partition.main([*base, '--arm', 'ahc_average_within_class'])


def test_partition_context_reports_both_folds_with_the_bars(tmp_path: Path) -> None:
    from scripts.research.cluster_diag import class_prior_run as run, heldout as ho

    stem, lab, gt, stored = _project(tmp_path)
    b = load_bundle(stem, lab, gt)
    ctx = run.PartitionContext(b, run.stored_partition(stored, b))
    first, _ = ho.split_groups(b.group, 5)
    report = ctx.split_report(ho.half_mask(b.group, first))
    folds = report['folds']
    cell = folds['fold1_calibration_to_heldout']['families']['within_detector_class']['0.1']
    rows = {r['metric']: r for r in cell['bars']}
    assert rows['bcubed_precision_all']['value'] == pytest.approx(1.0)
    assert rows['permutation_ari']['value'] == pytest.approx(1.0)
    assert rows['protected_items_moved']['value'] == 0
    assert not rows['bcubed_precision_low_res']['pass']  # the toy set has no low_res stratum
    assert set(folds) == {'fold1_calibration_to_heldout', 'fold2_heldout_to_calibration'}
    refs = folds['fold1_calibration_to_heldout']['references']
    assert refs['detector_class_only']['all']['bcubed_precision'] == pytest.approx(1.0)
    assert refs['all_singletons']['all']['bcubed_recall'] < 0.1


def test_stored_partition_aligns_to_the_labelled_crops_and_marks_missing_ones(
    tmp_path: Path,
) -> None:
    from scripts.research.cluster_diag import class_prior_run as run

    stem, lab, gt, stored = _project(tmp_path, per_class=3)
    rows = stored.read_text().splitlines()
    stored.write_text('\n'.join(rows[:-1]) + '\n')  # the last crop has no stored cluster
    b = load_bundle(stem, lab, gt)
    got = run.stored_partition(stored, b)
    assert got.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, -1]


def test_extra_split_manifests_are_seeded_disjoint_covers(tmp_path: Path) -> None:
    from scripts.research.cluster_diag import class_prior_run as run

    pin = tmp_path / 'pin_set.json'
    images = [{'file_name': f'{i}.jpg', 'group': 'g1' if i % 2 else 'g2'} for i in range(40)]
    pin.write_text(json.dumps({'images': images, 'manifest_sha256': 'x'}))
    a, again, other = (
        run._extra_manifest(pin, 1),
        run._extra_manifest(pin, 1),
        run._extra_manifest(pin, 2),
    )
    assert a['calibration'] == again['calibration']
    assert a['calibration'] != other['calibration']
    assert set(a['calibration']).isdisjoint(a['heldout'])
    assert len(a['calibration']) + len(a['heldout']) == 40


def test_the_scale_timing_run_records_cost_and_the_largest_cluster(tmp_path: Path) -> None:
    from scripts.research.cluster_diag import class_prior_run as run

    stem, _, _, _ = _project(tmp_path, per_class=20)
    for mode in ('dense', 'sparse'):
        out = run.run_scale_time(
            Namespace(bundle=stem, key='car', mode=mode, cut=0.3, sweep=[0.00001])
        )
        assert out['key_crops'] == 20
        assert out['clusters'] == 1  # one tight blob: a cut of 0.3 keeps it whole
        assert out['largest_cluster_share'] == 1.0
        assert out['crops_per_key'] == {'bus': 20, 'car': 20, 'dog': 20}
        assert out['fit_seconds'] >= 0
        if mode == 'dense':  # a cut below every merge height leaves every crop alone
            assert out['cut_sweep_same_dendrogram']['1e-05'] == {
                'clusters': 20,
                'largest_cluster_share': 1 / 20,
            }
        assert not math.isnan(out['peak_rss_gb'])


def test_every_result_file_names_the_commit_and_the_host_load(tmp_path: Path) -> None:
    from scripts.bench import scorecard
    from scripts.research.cluster_diag import class_prior_run as run

    stem, _, _, _ = _project(tmp_path, per_class=5)
    out = tmp_path / 'sub' / 'r.json'
    argv = ['scale-time', '--bundle', str(stem), '--out', str(out)]
    assert run.main([*argv, '--key', 'dog', '--mode', 'dense', '--cut', '0.3']) == 0
    got = json.loads(out.read_text())
    assert got['command'] == 'scale-time'
    assert got['git_sha'] == scorecard.current_git_sha()
    assert len(got['load_avg_start']) == len(got['load_avg_end']) == 3
    assert got['key'] == 'dog'
