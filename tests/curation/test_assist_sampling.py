"""Representative sampling is pure and method independent: the size grows with
the scope, strata each get a share, and a seed makes it reproducible."""

from __future__ import annotations

import random

import pytest

from src.domain.assist_sampling import SAMPLE_HARD_MAX, representative_size, stratified_sample


@pytest.mark.parametrize(
    ('n', 'expected'),
    [(0, 0), (5, 5), (8, 8), (10, 8), (25, 10), (100, 20), (400, 40), (1000, 64), (50000, 64)],
)
def test_default_size_table(n: int, expected: int) -> None:
    assert representative_size(n) == expected


def test_explicit_size_is_capped_by_the_scope() -> None:
    assert representative_size(100, size=30) == 30
    assert representative_size(20, size=30) == 20


@pytest.mark.parametrize('size', [0, -1, SAMPLE_HARD_MAX + 1])
def test_size_outside_the_hard_max_is_refused(size: int) -> None:
    with pytest.raises(ValueError, match='size'):
        representative_size(1000, size=size)


def _strata(sizes: dict[str, int]) -> dict[str, list[str]]:
    return {key: [f'{key}-{i:03d}' for i in range(n)] for key, n in sizes.items()}


def test_every_stratum_gets_at_least_one_when_k_allows() -> None:
    strata = _strata({'a': 100, 'b': 3, 'c': 1})
    picked = stratified_sample(strata, 10, seed=0)
    assert len(picked) == 10
    for key in strata:
        assert any(p.startswith(f'{key}-') for p in picked)


def test_largest_remainder_allocation_sums_to_k() -> None:
    strata = _strata({'a': 50, 'b': 30, 'c': 20})
    for k in (3, 7, 10, 33, 100):
        picked = stratified_sample(strata, k, seed=1)
        assert len(picked) == k
        assert len(set(picked)) == k
    counts = {
        key: sum(p.startswith(f'{key}-') for p in stratified_sample(strata, 10, seed=1))
        for key in strata
    }
    assert counts == {'a': 5, 'b': 3, 'c': 2}


def test_a_small_stratum_is_never_over_drawn() -> None:
    strata = _strata({'a': 2, 'b': 2, 'c': 6})
    picked = stratified_sample(strata, 9, seed=0)
    assert len(set(picked)) == 9
    assert sum(p.startswith('a-') for p in picked) == 2
    assert sum(p.startswith('c-') for p in picked) == 5


def test_fewer_picks_than_strata_take_the_largest_strata() -> None:
    strata = _strata({'a': 5, 'b': 50, 'c': 20, 'd': 20})
    picked = stratified_sample(strata, 2, seed=0)
    assert sorted(p.split('-')[0] for p in picked) == ['b', 'c']  # tie c/d by key


def test_k_above_the_total_returns_everything_sorted() -> None:
    strata = _strata({'a': 2, 'b': 3})
    assert stratified_sample(strata, 99, seed=0) == sorted(
        i for ids in strata.values() for i in ids
    )


def test_empty_strata_are_ignored() -> None:
    assert stratified_sample({'a': [], 'b': ['x']}, 5, seed=0) == ['x']
    assert stratified_sample({}, 5, seed=0) == []


def test_same_seed_same_ids_different_seed_different_ids() -> None:
    strata = _strata({'': 500})
    one = stratified_sample(strata, 20, seed=7)
    assert one == stratified_sample(strata, 20, seed=7)
    assert one != stratified_sample(strata, 20, seed=8)
    assert one == sorted(one)


def test_result_does_not_depend_on_input_order() -> None:
    strata = _strata({'a': 40, 'b': 25, 'c': 9})
    shuffled = {k: random.Random(3).sample(v, len(v)) for k, v in reversed(list(strata.items()))}
    assert stratified_sample(strata, 12, seed=5) == stratified_sample(shuffled, 12, seed=5)
