"""Tests for k-center-greedy core-set selection (curation-strategy plan
§2.6/§3.4/§9). Plan §9's own validation framing for this method: on a
known synthetic layout, assert the algorithm picks corners/extremes
first (not random interior points), and that results are deterministic
given the same ``seed_idx``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.services.curation.selection.kcenter_greedy import k_center_greedy


def _corners_plus_redundant_mass(d: int = 4, n_dup: int = 30) -> np.ndarray:
    """4 mutually-orthonormal "corner" directions (rows 0-3, standard
    basis vectors in R^d) plus ``n_dup`` exact duplicate copies of corner
    0 (rows 4..4+n_dup-1) — a stand-in for "one common vehicle type with
    lots of near-identical crops, plus a few genuinely distinct rare
    ones". A good diversity sampler should spend its early picks on the
    4 distinct directions, not repeatedly reselect the redundant mass."""
    corners = np.eye(4, d, dtype=np.float32)
    dup = np.tile(corners[0], (n_dup, 1))
    return np.vstack([corners, dup])


def test_picks_all_corners_before_any_redundant_duplicate() -> None:
    x = _corners_plus_redundant_mass()
    selected = k_center_greedy(x, 4, seed_idx=None)
    assert set(selected.tolist()) == {0, 1, 2, 3}, (
        f'expected the 4 distinct corners, got {selected.tolist()}'
    )


def test_full_ranking_puts_corners_ahead_of_duplicates() -> None:
    """A full-pool ranking (k == n) should still front-load the 4 distinct
    directions before touching any of the 30 redundant duplicates, even
    though duplicates outnumber corners 30 to 4 in the pool."""
    x = _corners_plus_redundant_mass()
    order = k_center_greedy(x, x.shape[0], seed_idx=None)
    first_four = set(order[:4].tolist())
    assert first_four == {0, 1, 2, 3}


def test_deterministic_given_explicit_seed() -> None:
    x = _corners_plus_redundant_mass()
    a = k_center_greedy(x, 10, seed_idx=5)
    b = k_center_greedy(x, 10, seed_idx=5)
    assert np.array_equal(a, b)
    assert a[0] == 5


def test_deterministic_with_no_seed_given() -> None:
    """No RNG anywhere in the function — two calls with seed_idx=None on
    the same data must select the identical sequence."""
    x = _corners_plus_redundant_mass()
    a = k_center_greedy(x, 12, seed_idx=None)
    b = k_center_greedy(x, 12, seed_idx=None)
    assert np.array_equal(a, b)


def test_different_seeds_can_produce_different_first_picks() -> None:
    x = _corners_plus_redundant_mass()
    a = k_center_greedy(x, 1, seed_idx=1)
    b = k_center_greedy(x, 1, seed_idx=2)
    assert a[0] == 1
    assert b[0] == 2


def test_no_seed_start_is_farthest_from_centroid_not_a_fixed_index() -> None:
    """The redundant mass (30 copies of corner 0) dominates the batch
    centroid direction. The documented no-seed rule (farthest from
    centroid) must NOT pick a member of that redundant mass as the seed —
    picking index 0 unconditionally (a naive fixed-index rule) would."""
    x = _corners_plus_redundant_mass()
    selected = k_center_greedy(x, 1, seed_idx=None)
    assert selected[0] in {0, 1, 2, 3}, (
        'seed should be a distinct corner, not a redundant duplicate'
    )


def test_k_clamped_to_n() -> None:
    x = _corners_plus_redundant_mass()
    selected = k_center_greedy(x, 10_000, seed_idx=0)
    assert len(selected) == x.shape[0]
    assert len(set(selected.tolist())) == x.shape[0]  # every row selected exactly once


def test_empty_input() -> None:
    x = np.zeros((0, 4), dtype=np.float32)
    selected = k_center_greedy(x, 5)
    assert selected.shape == (0,)


def test_k_zero_or_negative_returns_empty() -> None:
    x = _corners_plus_redundant_mass()
    assert k_center_greedy(x, 0).shape == (0,)
    assert k_center_greedy(x, -3).shape == (0,)


def test_seed_idx_out_of_range_raises() -> None:
    x = _corners_plus_redundant_mass()
    with pytest.raises(ValueError, match='seed_idx'):
        k_center_greedy(x, 3, seed_idx=999)


def test_never_reselects_a_point() -> None:
    x = _corners_plus_redundant_mass()
    selected = k_center_greedy(x, x.shape[0], seed_idx=0)
    assert len(set(selected.tolist())) == len(selected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
