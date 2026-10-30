"""Tests for the standard-benchmark scorer: metrics against references, protocol counts, bar verdicts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from scripts.research.standard_benchmarks import bars, embed_run, metrics, protocol, score_run


if TYPE_CHECKING:
    from pathlib import Path


def _unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _blobs(
    per: int, centres: int, seed: int = 0, noise: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = _unit(rng, centres, 32)
    x = np.concatenate([c[i] + noise * rng.normal(size=(per, 32)) for i in range(centres)])
    y = np.repeat(np.arange(centres), per)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), y


def test_recall_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    x = _unit(rng, 60, 8)
    y = rng.integers(0, 9, 60)
    ks = (1, 2, 4, 8)
    got = metrics.recall_hits(x, y, ks)
    sim = x @ x.T
    for i in range(60):
        order = [j for j in np.argsort(-sim[i], kind='stable') if j != i]
        for c, k in enumerate(ks):
            assert got[i, c] == any(y[j] == y[i] for j in order[:k])


def test_recall_excludes_the_query_but_counts_a_duplicate() -> None:
    x = np.array([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0], [0.7, 0.7]], dtype=np.float32)
    y = np.array([0, 0, 1, 2, 3])
    hits = metrics.recall_hits(x, y, (1,))[:, 0]
    assert hits.tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]  # row 3 sits on row 2 but another class


def test_recall_rejects_k_not_below_rows() -> None:
    with pytest.raises(ValueError, match='more than'):
        metrics.recall_hits(np.eye(3, dtype=np.float32), np.array([0, 0, 1]), (3,))


def test_nmi_matches_scikit_learn_and_bounds() -> None:
    rng = np.random.default_rng(2)
    a, b = rng.integers(0, 7, 300), rng.integers(0, 5, 300)
    assert metrics.nmi(a, b) == pytest.approx(normalized_mutual_info_score(a, b), abs=1e-9)
    assert metrics.nmi(a, a) == pytest.approx(1.0)
    assert metrics.nmi(a, (a * 3 + 1) % 50) == pytest.approx(1.0)  # relabelling does not matter
    assert metrics.nmi(np.zeros(10, dtype=int), np.zeros(10, dtype=int)) == 1.0


def test_kmeans_recovers_separable_blobs_and_is_seeded() -> None:
    x, y = _blobs(40, 6)
    a = metrics.kmeans(x, 6, seed=0)
    assert metrics.nmi(y, a) == pytest.approx(1.0)
    assert np.array_equal(a, metrics.kmeans(x, 6, seed=0))
    sk = KMeans(6, n_init=3, random_state=0).fit_predict(x)
    assert metrics.nmi(y, a) == pytest.approx(normalized_mutual_info_score(y, sk))


def test_kmeans_lloyd_steps_improve_on_the_seeding() -> None:
    x, y = _blobs(60, 8, seed=4, noise=0.12)
    one_seed = [metrics.nmi(y, metrics.kmeans(x, 8, s, max_iter=300)) for s in range(4)]
    sk = [
        normalized_mutual_info_score(y, KMeans(8, n_init=1, random_state=s).fit_predict(x))
        for s in range(4)
    ]
    assert np.mean(one_seed) >= np.mean(sk) - 0.05
    assert np.mean(one_seed) > 0.7
    # converged: every point is at least as near its own cluster mean as to any other mean
    x2, _ = _blobs(50, 6, seed=7, noise=0.4)
    a = metrics.kmeans(x2, 6, seed=0)
    means = np.stack([x2[a == c].mean(0) for c in range(6)])
    nearest = ((x2[:, None, :] - means[None]) ** 2).sum(2).argmin(1)
    assert (nearest == a).mean() > 0.99


def test_kmeans_with_k_of_one_puts_everything_together() -> None:
    x, _ = _blobs(10, 3)
    assert set(metrics.kmeans(x, 1, seed=0).tolist()) == {0}


def test_paired_bootstrap_on_a_constant_and_a_null_difference() -> None:
    a = np.ones(200, dtype=bool)
    b = np.zeros(200, dtype=bool)
    iv = metrics.paired_bootstrap(a, b, resamples=200)
    assert (iv['delta'], iv['lo'], iv['hi']) == (1.0, 1.0, 1.0)
    rng = np.random.default_rng(3)
    same = rng.random(400) < 0.5
    null = metrics.paired_bootstrap(same, same, resamples=200)
    assert null['lo'] == null['hi'] == 0.0
    shifted = metrics.paired_bootstrap(same | (rng.random(400) < 0.2), same, resamples=500)
    assert shifted['lo'] > 0
    assert shifted['lo'] < shifted['delta'] < shifted['hi']


def _raw(n_classes: int, per: int, split: str) -> list[dict[str, object]]:
    return [
        {'path': f'{split}/{c}/{i}.jpg', 'item_id': f'{split}{c}', 'split': split}
        for c in range(n_classes)
        for i in range(per)
    ]


def test_sample_products_is_seeded_and_bounded() -> None:
    raw = _raw(50, 2, 'train')
    a = protocol.sample_products(raw, 'item_id', 10)
    assert a == protocol.sample_products(raw, 'item_id', 10)
    assert len(a) == 10
    assert a != protocol.sample_products(raw, 'item_id', 10, seed=1)
    assert len(protocol.sample_products(raw, 'item_id', 500)) == 50


def test_select_rows_splits_by_the_registered_key() -> None:
    spec = protocol.SetSpec('d', 'item_id', 'split', 'test', 'train', 4, 2, 8)
    rows = protocol.select_rows(_raw(2, 2, 'test') + _raw(2, 2, 'train'), spec, sop_sample=False)
    assert [r.part for r in rows] == ['test'] * 4 + ['calib'] * 4
    assert all(r.label.startswith(r.part.replace('calib', 'train')) for r in rows)


def test_load_rows_stops_on_a_wrong_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = protocol.SetSpec('d', 'item_id', 'split', 'test', 'train', 4, 2, 8)
    monkeypatch.setitem(protocol.SETS, 'toy', spec)
    (tmp_path / 'd').mkdir()
    raw = _raw(2, 2, 'test') + _raw(2, 2, 'train')
    (tmp_path / 'd' / 'labels.jsonl').write_text('\n'.join(json.dumps(r) for r in raw))
    assert len(protocol.load_rows(tmp_path, 'toy')) == 8
    (tmp_path / 'd' / 'labels.jsonl').write_text('\n'.join(json.dumps(r) for r in raw[:-1]))
    with pytest.raises(protocol.ProtocolError, match='registered 8'):
        protocol.load_rows(tmp_path, 'toy')
    moved = [dict(r) for r in raw]
    moved[0]['split'] = 'train'  # total unchanged, test part one image short
    (tmp_path / 'd' / 'labels.jsonl').write_text('\n'.join(json.dumps(r) for r in moved))
    with pytest.raises(protocol.ProtocolError, match='test part has 3'):
        protocol.load_rows(tmp_path, 'toy')
    wrong_classes = _raw(1, 4, 'test') + _raw(2, 2, 'train')
    (tmp_path / 'd' / 'labels.jsonl').write_text('\n'.join(json.dumps(r) for r in wrong_classes))
    with pytest.raises(protocol.ProtocolError, match='in 1 classes'):
        protocol.load_rows(tmp_path, 'toy')


def test_sop_cluster_subset_keeps_whole_products(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, 'SOP_CLUSTER_PRODUCTS', 5)
    rows = [protocol.Row(f'{p}/{i}', f'p{p}', 'test') for p in range(20) for i in range(3)]
    idx = protocol.cluster_subset(rows, 'sop', 'test')
    labels = [rows[i].label for i in idx]
    assert len(set(labels)) == 5
    assert len(idx) == 15
    assert len(protocol.cluster_subset(rows, 'cars196', 'test')) == 60


def _iv(delta: float, half: float) -> dict[str, float]:
    return {'delta': delta, 'lo': delta - half, 'hi': delta + half}


def test_gain_verdict_table() -> None:
    ok = {'cub': _iv(0.0, 0.005), 'sop': _iv(0.0, 0.005)}
    assert bars.gain_verdict(_iv(0.05, 0.01), ok) == bars.PASS
    assert bars.gain_verdict(_iv(0.05, 0.06), ok) == bars.INCONCLUSIVE  # interval reaches 0
    assert bars.gain_verdict(_iv(0.02, 0.005), ok) == bars.FAIL  # too small, confidently
    assert (
        bars.gain_verdict(_iv(0.02, 0.03), ok) == bars.INCONCLUSIVE
    )  # could still reach the margin
    worse = {'cub': _iv(-0.05, 0.01), 'sop': _iv(0.0, 0.005)}
    assert bars.gain_verdict(_iv(0.08, 0.01), worse) == bars.FAIL  # a regression elsewhere
    mild = {'cub': _iv(-0.02, 0.03), 'sop': _iv(0.0, 0.005)}
    assert bars.gain_verdict(_iv(0.08, 0.01), mild) == bars.INCONCLUSIVE


def test_cut_verdict_needs_a_calibration_point_and_test_precision() -> None:
    point = {'param': 0.3, 'precision': 0.92, 'recall': 0.1}
    assert bars.cut_verdict(point, 0.90) == bars.PASS
    assert bars.cut_verdict(point, 0.8999) == bars.FAIL
    assert bars.cut_verdict(None, 1.0) == bars.FAIL


def _model(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        'recall': {'1': 0.5, '2': 0.6, '4': 0.7, '8': 0.8},
        'control_recall1': 0.01,
        'control_nmi': 0.01,
        'nmi': 0.5,
        'rows': 8131,
        'expected_rows': 8131,
        'all_recall_max': 0.7,
    }
    return {**base, **over}


def test_sanity_flags_each_failure_separately() -> None:
    res = bars.sanity({'a': _model()})  # type: ignore[arg-type]
    assert res['all_pass'] is True
    cases = {
        'control_is_chance': _model(control_recall1=0.5),
        'recall_monotone': _model(recall={'1': 0.5, '2': 0.4, '4': 0.7, '8': 0.8}),
        'nmi_in_range': _model(nmi=1.0),
        'rows_match': _model(rows=8130),
        'recall1_plausible': _model(all_recall_max=0.97),
    }
    for key, model in cases.items():
        got = bars.sanity({'a': model})  # type: ignore[arg-type]
        assert got['a'][key] is False, key
        assert got['all_pass'] is False, key
    low = bars.sanity({'a': _model(recall={'1': 0.1, '2': 0.6, '4': 0.7, '8': 0.8})})  # type: ignore[arg-type]
    assert low['a']['recall1_plausible'] is False


def test_weak_ordering_and_contradictions() -> None:
    r1 = dict(zip(bars.SETC_P10, [0.9, 0.8, 0.7, 0.6, 0.5], strict=True))
    assert bars.weak_ordering({**r1, 'siglip2_so400m_p16_256': 0.7, 'dinov2_l14': 0.7})[
        'consistent'
    ]
    assert not bars.weak_ordering({**r1, 'siglip2_so400m_p16_256': 0.6, 'dinov2_l14': 0.7})[
        'consistent'
    ]
    # set C ranks PE-G above DINOv2; here DINOv2 is clearly higher on CARS196
    iv = {('dinov2_l14', 'pe_core_g14_448'): _iv(0.1, 0.01)}
    found = bars.ordering_contradictions(r1, iv)
    assert [(f['set_c_higher'], f['cars_higher']) for f in found] == [
        ('pe_core_g14_448', 'dinov2_l14')
    ]
    # an interval that straddles 0 is not a contradiction; the same order as set C is none
    assert (
        bars.ordering_contradictions(r1, {('dinov2_l14', 'pe_core_g14_448'): _iv(0.1, 0.2)}) == []
    )
    assert (
        bars.ordering_contradictions(r1, {('pe_core_g14_448', 'dinov2_l14'): _iv(0.1, 0.01)}) == []
    )


def test_partition_scores_and_ahc_reading_on_separable_data() -> None:
    xc, yc = _blobs(12, 8, seed=1)
    xt, yt = _blobs(12, 10, seed=2)
    perfect = score_run.partition_scores(yt, yt)
    assert perfect['ari'] == pytest.approx(1.0)
    assert perfect['bcubed_f'] == pytest.approx(1.0)
    one = score_run.partition_scores(yt, np.zeros_like(yt))
    assert one['bcubed_precision'] < 0.2
    assert one['ari'] == pytest.approx(0.0, abs=1e-9)
    res = score_run.ahc_reading(xc, yc, xt, yt)
    assert res['cut_at_k']['clusters'] == 10
    assert res['cut_at_k']['ari'] == pytest.approx(1.0)
    assert res['operating_point'] is not None
    assert res['operating_cut']['bcubed_precision'] == pytest.approx(1.0)
    assert res['verdict_b3'] == bars.PASS


def test_ahc_reading_fails_when_no_calibration_cut_is_precise() -> None:
    rng = np.random.default_rng(5)
    xc = _unit(rng, 80, 16)  # no structure
    yc = rng.integers(0, 4, 80)
    xt, yt = _blobs(10, 5, seed=3)
    res = score_run.ahc_reading(xc, yc, xt, yt)
    assert res['operating_point'] is None
    assert res['operating_cut'] is None
    assert res['verdict_b3'] == bars.FAIL


def test_squares_cut_centre_and_letterbox_from_one_decode(tmp_path: Path) -> None:
    path = tmp_path / 'wide.png'
    arr = np.zeros((40, 80, 3), dtype=np.uint8)
    arr[:, :15] = (255, 0, 0)  # left strip red: lost by the centre crop, kept by the letterbox
    Image.fromarray(arr).save(path)
    ds = embed_run.Squares([path], [32, 64], ['centre', 'letterbox'])
    item = ds[0]
    assert set(item) == {'32_centre', '32_letterbox', '64_centre', '64_letterbox'}
    assert item['64_centre'].shape == (64, 64, 3)
    assert item['32_letterbox'].shape == (32, 32, 3)
    assert item['64_centre'][:, :, 0].max() == 0  # the red strip is cropped away
    lb = item['64_letterbox']
    assert lb[:, :12, 0].max() == 255  # kept on the left
    assert (lb[0, 0] == 128).all()
    assert (lb[-1, -1] == 128).all()
    batch = embed_run._collate([ds[0], ds[0]])
    assert batch['32_centre'].shape == (2, 32, 32, 3)


def test_best_variant_follows_calibration_and_ties_keep_centre() -> None:
    combos = {
        'm__centre': {'calibration_recall1': 0.5},
        'm__letterbox': {'calibration_recall1': 0.6},
        'n__centre': {'calibration_recall1': 0.5},
        'n__letterbox': {'calibration_recall1': 0.5},
    }
    assert score_run.best_variant(combos, 'm') == 'letterbox'
    assert score_run.best_variant(combos, 'n') == 'centre'


def test_recall_copies_mask_hides_identical_files_only() -> None:
    x = np.array([[1.0, 0], [1.0, 0], [0.9, 0.436], [0, 1.0]], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.array([0, 0, 1, 1])
    assert metrics.recall_hits(x, y, (1,))[:, 0].tolist() == [True, True, False, True]
    masked = metrics.recall_hits(x, y, (1,), copies=np.array([7, 7, 8, 9]))[:, 0]
    assert masked.tolist() == [
        False,
        False,
        False,
        True,
    ]  # rows 0 and 1 lose each other; row 3 is unaffected


def test_file_groups_reads_manifest_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = protocol.SetSpec('d', 'item_id', 'split', 'test', 'train', 1, 1, 2)
    monkeypatch.setitem(protocol.SETS, 'toy', spec)
    (tmp_path / 'd').mkdir()
    files = [{'path': 'a.jpg', 'sha256': 'h1'}, {'path': 'b.jpg', 'sha256': 'h1'}]
    (tmp_path / 'd' / 'manifest.json').write_text(json.dumps({'files': files}))
    rows = [protocol.Row('b.jpg', 'x', 'test'), protocol.Row('a.jpg', 'x', 'test')]
    assert protocol.file_groups(tmp_path, 'toy', rows) == ['h1', 'h1']


def test_top1_similarity_finds_the_nearest_other_row() -> None:
    x = np.array([[1.0, 0], [1.0, 0], [0, 1.0]], dtype=np.float32)
    assert metrics.top1_similarity(x).tolist() == [1.0, 1.0, 0.0]
