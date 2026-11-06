"""The pure grouping rules: components, decisions, the size guard, representatives, lineage."""

from __future__ import annotations

import numpy as np
import pytest

from src.domain import near_dup as rules


def _edges(*pairs: tuple[str, str, float]) -> list[tuple[str, str, float]]:
    return list(pairs)


def test_components_are_transitive_and_sorted() -> None:
    eff = rules.effective_edges(_edges(('c', 'b', 0.99), ('a', 'b', 0.99), ('x', 'y', 0.995)), 0.98)
    assert rules.connected_components(eff) == [['a', 'b', 'c'], ['x', 'y']]


def test_an_edge_below_the_cut_does_not_link() -> None:
    eff = rules.effective_edges(_edges(('a', 'b', 0.979), ('b', 'c', 0.98)), 0.98)
    assert rules.connected_components(eff) == [['b', 'c']]


def test_the_result_does_not_depend_on_edge_order() -> None:
    edges = _edges(('a', 'b', 0.99), ('b', 'c', 0.985), ('d', 'e', 0.99), ('c', 'd', 0.97))
    forward = rules.group_members(rules.effective_edges(edges, 0.98), 1000)
    backward = rules.group_members(rules.effective_edges(edges[::-1], 0.98), 1000)
    assert forward == backward


def test_different_removes_an_edge_and_same_adds_one() -> None:
    edges = _edges(('a', 'b', 0.99), ('c', 'd', 0.5))
    decisions = {('a', 'b'): 'different', ('c', 'd'): 'same'}
    eff = rules.effective_edges(edges, 0.98, decisions)
    assert rules.connected_components(eff) == [['c', 'd']]


def test_a_decision_given_in_either_order_binds() -> None:
    eff = rules.effective_edges(_edges(('a', 'b', 0.99)), 0.98, {('b', 'a'): 'different'})
    assert eff == {}


def test_an_oversized_component_is_split_by_complete_linkage() -> None:
    # a chain a-b-c-d-e: every neighbour pair links, so one component of five.
    chain = _edges(('a', 'b', 0.99), ('b', 'c', 0.99), ('c', 'd', 0.99), ('d', 'e', 0.99))
    eff = rules.effective_edges(chain, 0.98)
    whole = rules.group_members(eff, 1000)
    assert whole.groups == [['a', 'b', 'c', 'd', 'e']]
    assert whole.oversized_split == 0
    split = rules.group_members(eff, 3)
    assert split.oversized_split == 1
    # complete linkage: members of one group are all directly linked; a chain only gives pairs.
    assert all(len(g) == 2 for g in split.groups)
    assert all(rules.order_pair(g[0], g[1]) in eff for g in split.groups)


def test_a_clique_within_the_guard_is_kept_whole_when_split() -> None:
    clique = [(a, b, 0.99) for a, b in (('a', 'b'), ('a', 'c'), ('b', 'c'), ('d', 'e'))]
    eff = rules.effective_edges(clique, 0.98)
    assert rules.group_members(eff, 3).groups == [['a', 'b', 'c'], ['d', 'e']]


def test_the_representative_is_the_most_central_and_ties_go_to_the_smaller_id() -> None:
    mat = np.array([[1, 0], [0.9, 0.1], [0.99, 0.01]], dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    group = rules.describe_group(['a', 'b', 'c'], mat, {('a', 'b'): 0.99, ('a', 'c'): 0.99})
    assert group.representative == 'c'
    same = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (3, 1))
    assert rules.describe_group(['x', 'y', 'z'], same, {}).representative == 'x'
    assert group.cosine_to_rep['c'] == pytest.approx(1.0)
    assert group.min_cosine_to_rep < 1.0


def test_lineage_is_carried_at_jaccard_one_half_and_used_once() -> None:
    parent = {'p1': ('L1', {'a', 'b', 'c', 'd'}), 'p2': ('L2', {'x', 'y'})}
    new = {'n1': {'a', 'b', 'c'}, 'n2': {'a', 'q'}, 'n3': {'x', 'y', 'z'}, 'n4': {'m', 'n'}}
    carried = rules.assign_lineage(new, parent)
    assert carried['n1'] == 'L1'  # 3/4
    assert carried['n2'] is None  # 1/5, and L1 is taken
    assert carried['n3'] == 'L2'  # 2/3
    assert carried['n4'] is None
