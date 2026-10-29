"""Node paths, seeded child sampling and best-first order for the auto-label loop."""

from __future__ import annotations

import random

import pytest

from src.domain.loop_tree import (
    best_first_key,
    child_path,
    child_sample,
    parent_path,
    parse_path,
    path_level,
    root_path,
    seed_int,
)


def test_path_grammar_round_trips() -> None:
    assert root_path(10042) == 'r10042'
    assert child_path('r10042', 2) == 'r10042.2'
    assert child_path('r10042.2', 0) == 'r10042.2.0'
    assert parse_path('r10042') == (10042, ())
    assert parse_path('r10042.2.0') == (10042, (2, 0))
    for path in ('r10000', 'r10042.2', 'r10042.2.0.11'):
        root, indices = parse_path(path)
        rebuilt = root_path(root)
        for index in indices:
            rebuilt = child_path(rebuilt, index)
        assert rebuilt == path


def test_levels_and_parents() -> None:
    assert path_level('r7') == 0
    assert path_level('r7.1.3') == 2
    assert parent_path('r7.1.3') == 'r7.1'
    assert parent_path('r7.1') == 'r7'
    assert parent_path('r7') is None


@pytest.mark.parametrize(
    'bad', ['', 'x1', 'r', 'r1.', 'r1..2', 'r1.a', 'r-1', 'r1.-2', 'R1', ' r1']
)
def test_bad_paths_are_refused(bad: str) -> None:
    with pytest.raises(ValueError, match='path'):
        parse_path(bad)


def test_bad_child_index_is_refused() -> None:
    with pytest.raises(ValueError, match='index'):
        child_path('r1', -1)


IDS = [f'c{i:04d}' for i in range(400)]


def test_child_sample_reuses_the_parents_answers() -> None:
    child = IDS[:100]
    parent_sampled = [IDS[3], IDS[50], IDS[250], IDS[399]]
    got = child_sample(parent_sampled, child, 10, seed=7, path='r1.0')
    assert got.reused == sorted([IDS[3], IDS[50]])
    assert len(got.draw) == 8
    assert not set(got.draw) & set(got.reused)
    assert set(got.draw) <= set(child)
    assert got.draw == sorted(got.draw)


def test_child_sample_never_draws_more_than_exists() -> None:
    child = IDS[:5]
    got = child_sample([IDS[0]], child, 50, seed=1, path='r1.0')
    assert got.reused == [IDS[0]]
    assert got.draw == sorted(IDS[1:5])


def test_child_sample_keeps_every_reused_answer_above_k() -> None:
    child = IDS[:100]
    parent_sampled = IDS[:30]
    got = child_sample(parent_sampled, child, 10, seed=1, path='r1.0')
    assert got.reused == IDS[:30]
    assert got.draw == []


def test_same_seed_and_path_give_the_same_ids_whatever_the_order() -> None:
    child = IDS[:200]
    shuffled = list(child)
    random.Random(3).shuffle(shuffled)
    parent = [IDS[1], IDS[9]]
    first = child_sample(parent, child, 30, seed=5, path='r1.2')
    second = child_sample(list(reversed(parent)), shuffled, 30, seed=5, path='r1.2')
    assert first == second


def test_seed_and_path_change_the_draw() -> None:
    child = IDS[:300]
    base = child_sample([], child, 30, seed=5, path='r1.2')
    assert child_sample([], child, 30, seed=6, path='r1.2').draw != base.draw
    assert child_sample([], child, 30, seed=5, path='r1.3').draw != base.draw


def test_root_draw_has_no_reuse() -> None:
    got = child_sample([], IDS[:50], 12, seed=0, path='r4')
    assert got.reused == []
    assert len(got.draw) == 12


def test_child_sample_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match='k must'):
        child_sample([], IDS[:5], 0, seed=0, path='r1')


def test_best_first_orders_by_size_then_numeric_path() -> None:
    nodes = [(50, 'r2'), (900, 'r10.1'), (900, 'r10.10'), (900, 'r10.2'), (900, 'r3')]
    ordered = sorted(nodes, key=lambda node: best_first_key(*node))
    assert ordered == [
        (900, 'r3'),
        (900, 'r10.1'),
        (900, 'r10.2'),
        (900, 'r10.10'),
        (50, 'r2'),
    ]


def test_seed_int_is_stable_and_depends_on_seed_and_path() -> None:
    first = seed_int(5, 'r1.2')
    assert first == seed_int(5, 'r1.2')
    assert 0 <= first < 2**32
    assert seed_int(6, 'r1.2') != first
    assert seed_int(5, 'r1.3') != first
