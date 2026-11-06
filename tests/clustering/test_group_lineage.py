"""Jaccard + Hungarian carry-over: identities follow members, never numbering or order."""

from __future__ import annotations

import random

from src.domain.group_lineage import match_groups


def _groups(prefix: str, sizes: list[int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    at = 0
    for i, size in enumerate(sizes):
        out[f'{prefix}{i}'] = [f'item{n}' for n in range(at, at + size)]
        at += size
    return out


def test_equal_members_keep_identity_whatever_the_numbering() -> None:
    old = _groups('o', [30, 20, 10])
    shuffled = list(old.items())
    random.Random(7).shuffle(shuffled)
    new = {f'n{i}': members[::-1] for i, (_, members) in enumerate(shuffled)}
    result = match_groups(old, new)
    for i, (old_key, _) in enumerate(shuffled):
        lineage = result.by_new[f'n{i}']
        assert lineage.transition == 'survived'
        assert lineage.inherited == old_key
        assert lineage.jaccard == 1.0
    assert result.dropped == ()


def test_an_old_identity_goes_to_one_new_group_only() -> None:
    old = {'a': [str(i) for i in range(10)]}
    new = {'n0': [str(i) for i in range(5)], 'n1': [str(i) for i in range(5, 10)]}
    result = match_groups(old, new)  # both halves have Jaccard exactly 0.5
    kinds = sorted(lineage.transition for lineage in result.by_new.values())
    assert kinds == ['split', 'survived']
    assert sum(lin.inherited == 'a' for lin in result.by_new.values()) == 1


def test_below_the_threshold_is_a_new_group() -> None:
    old = {'a': ['1', '2', '3', '4']}
    new = {'n': ['1', 'x', 'y', 'z', 'w']}
    result = match_groups(old, new)
    assert result.by_new['n'].transition == 'new'
    assert result.dropped == ('a',)


def test_split_merge_and_absorb_are_classified() -> None:
    old = {
        's': [f's{i}' for i in range(20)],  # splits in two
        'm1': [f'm{i}' for i in range(10)],  # merges with m2
        'm2': [f'q{i}' for i in range(10)],
        'big': [f'b{i}' for i in range(40)],  # absorbs 'small'
        'small': [f'c{i}' for i in range(4)],
    }
    new = {
        'split_a': [f's{i}' for i in range(10)],
        'split_b': [f's{i}' for i in range(12, 20)],
        'merged': [f'm{i}' for i in range(10)]
        + [f'q{i}' for i in range(10)]
        + [f'z{i}' for i in range(6)],
        'bigger': [f'b{i}' for i in range(40)] + [f'c{i}' for i in range(4)],
    }
    result = match_groups(old, new)
    assert result.by_new['split_a'].transition == 'survived'
    assert result.by_new['split_b'].transition == 'split'
    assert result.by_new['split_b'].predecessors == ('s',)
    assert result.by_new['split_b'].inherited is None
    assert result.by_new['merged'].transition == 'merged'
    assert set(result.by_new['merged'].predecessors) == {'m1', 'm2'}
    absorbed = result.by_new['bigger']
    assert absorbed.transition == 'absorbed'
    assert absorbed.predecessors == ('big', 'small')
    assert absorbed.inherited == 'big'
    assert result.dropped == ()


def test_large_grouping_is_solved_per_component() -> None:
    old = _groups('o', [5] * 3000)
    new = {f'n{k[1:]}': list(v) for k, v in old.items()}
    result = match_groups(old, new)
    assert sum(1 for lin in result.by_new.values() if lin.transition == 'survived') == 3000
