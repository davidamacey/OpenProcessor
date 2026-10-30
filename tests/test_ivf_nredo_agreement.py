"""The nredo agreement gate (scripts/bench/ivf_nredo_agreement.py)."""

from __future__ import annotations

import numpy as np

from scripts.bench import ivf_nredo_agreement as gate
from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore


def _blobs(n: int = 600, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((12, dim))
    x = centers[rng.integers(0, 12, n)] + 0.3 * rng.standard_normal((n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_production_seed_is_the_store_fit(tmp_path) -> None:
    x = _blobs()
    ours = gate.train_centroids(x, n_clusters=16, niter=10, nredo=2, seed=gate.PRODUCTION_SEED)
    theirs, _ = IVFCentroidStore(tmp_path).train(x, n_clusters=16, niter=10, nredo=2, use_gpu=False)
    assert np.array_equal(ours, theirs)


def test_identical_partitions_agree_fully_and_a_relabelling_is_not_a_move() -> None:
    labels = np.random.default_rng(2).integers(0, 9, 400)
    same = gate.agreement(labels, (labels + 5) % 9 + 100)
    assert same['ari'] == 1.0
    assert same['same_cluster_share'] == 1.0


def test_unrelated_partitions_do_not_agree() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 9, 900), rng.integers(0, 9, 900)
    got = gate.agreement(a, b)
    assert got['ari'] < 0.05
    assert got['same_cluster_share'] < 0.4


def test_the_gate_refuses_a_candidate_that_loses_more_than_half_a_point() -> None:
    inc = {'purity': 0.80, 'false_merge_rate': 0.20}
    ok = {'purity': 0.796, 'false_merge_rate': 0.204}
    worse_purity = {'purity': 0.794, 'false_merge_rate': 0.20}
    worse_merge = {'purity': 0.80, 'false_merge_rate': 0.206}
    assert gate.judge(inc, ok, ari=0.9, ari_floor=0.9)['passes']
    assert not gate.judge(inc, worse_purity, ari=0.9, ari_floor=0.9)['passes']
    assert not gate.judge(inc, worse_merge, ari=0.9, ari_floor=0.9)['passes']


def test_the_gate_refuses_agreement_below_the_seed_noise_floor() -> None:
    inc = {'purity': 0.8, 'false_merge_rate': 0.2}
    assert gate.judge(inc, inc, ari=0.70, ari_floor=0.80)['checks'] == {
        'ari_not_below_noise_floor': False,
        'purity_within_half_point': True,
        'false_merge_within_half_point': True,
    }


def test_without_labels_only_agreement_decides() -> None:
    verdict = gate.judge({}, {}, ari=0.9, ari_floor=0.9)
    assert verdict['passes']
    assert verdict['checks']['purity_within_half_point'] is None


def _row(purity: float, inc_purity: float, ari: float = 0.33) -> dict:
    return {
        'labels': {'purity': purity, 'false_merge_rate': 0.3},
        'incumbent_same_seed_labels': {'purity': inc_purity, 'false_merge_rate': 0.3},
        'vs_incumbent_seed42': {'ari': ari},
    }


def test_seed_noise_that_cancels_in_the_mean_passes_but_a_real_loss_does_not() -> None:
    noisy = [_row(0.80, 0.79), _row(0.79, 0.80), _row(0.795, 0.795)]
    assert gate.judge_mean(noisy, ari_floor=0.33)['passes']
    lossy = [_row(0.78, 0.80), _row(0.78, 0.80)]
    assert not gate.judge_mean(lossy, ari_floor=0.33)['passes']
