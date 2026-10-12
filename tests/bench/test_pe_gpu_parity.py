from __future__ import annotations

import numpy as np
import pytest

from scripts.bench.pe_gpu_parity import pair_stats


def test_identical_vectors_are_counted_as_identical() -> None:
    v = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    stats = pair_stats(v, v.copy())
    assert stats['identical'] == 5
    assert stats['max_abs_diff'] == 0
    assert stats['cosine_min'] == pytest.approx(1.0)


def test_one_perturbed_crop_lowers_the_minimum_cosine_only() -> None:
    v = np.random.default_rng(1).normal(size=(4, 16)).astype(np.float32)
    w = v.copy()
    w[2, 0] += 1.0
    stats = pair_stats(v, w)
    assert stats['identical'] == 3
    assert stats['cosine_min'] < 1.0
    assert stats['cosine_mean'] > stats['cosine_min']


def test_different_crop_counts_are_refused() -> None:
    with pytest.raises(ValueError, match='not the same crops'):
        pair_stats(np.zeros((3, 4)), np.zeros((2, 4)))
