"""nd:P0: grouping rules, metrics and the registered bars on hand-built vectors (arithmetic in the comments)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.near_dup import group, sweep


def _unit(rows: list[list[float]]) -> np.ndarray:
    a = np.array(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _chain() -> np.ndarray:
    # Four unit vectors on a circle 10 degrees apart: neighbours cos(10 deg) = 0.985,
    # two apart cos(20 deg) = 0.940, ends cos(30 deg) = 0.866.
    angles = np.radians([0, 10, 20, 30])
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def test_single_linkage_chains_and_complete_does_not() -> None:
    x = _chain()
    edges = group.edges_from_vectors(x, floor=0.85)
    single, _ = group.group_labels(x, edges, 0.98, 'single')
    complete, _ = group.group_labels(x, edges, 0.98, 'complete')
    assert len(set(single.tolist())) == 1  # 0.985 links every neighbour: one component of four
    # complete linkage keeps {0,1} and {2,3} (each pair 0.985); no pair across the middle reaches 0.98
    assert sorted(len(np.flatnonzero(complete == g)) for g in set(complete.tolist())) == [2, 2]


def test_guard_splits_only_oversized_components() -> None:
    x = _chain()
    edges = group.edges_from_vectors(x, floor=0.85)
    big, n_big = group.group_labels(x, edges, 0.98, 'single_guard', guard=3)
    small, n_small = group.group_labels(x, edges, 0.98, 'single_guard', guard=4)
    assert n_big == 1
    assert len(set(big.tolist())) > 1
    assert n_small == 0
    assert len(set(small.tolist())) == 1


def test_matches_the_product_partition() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 16)).astype(np.float32)
    x = np.concatenate([base, base + 0.02 * rng.normal(size=base.shape).astype(np.float32)])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    edges = group.edges_from_vectors(x, floor=0.85)
    labels, _ = group.group_labels(x, edges, 0.95, 'single')
    mine = {frozenset(np.flatnonzero(labels == g).tolist()) for g in np.unique(labels)}
    mine = {g for g in mine if len(g) > 1}
    theirs = {frozenset(g) for g in group.product_groups(x, 0.95)}
    assert mine == theirs
    assert mine  # the noisy copies really group


def test_evaluate_recall_and_false_groups() -> None:
    # families 0 and 1; rows: orig0, orig1, q60 of 0, q60 of 1. Labels merge the two originals (a false group)
    # and leave copy of 1 alone.
    family = np.array([0, 1, 0, 1])
    transform = np.array(['orig', 'orig', 'T-q60', 'T-q60'])
    labels = np.array([0, 0, 0, 1])
    m = group.evaluate(labels, family, transform, {0: 0, 1: 1})
    assert (
        m['per_transform']['T-q60']['recovered'] == 1
    )  # copy of 0 shares a group with orig0; copy of 1 does not
    assert m['groups'] == 1
    assert m['false_groups'] == 1
    assert m['false_group_rate'] == 1.0
    assert m['largest_group'] == 3


def test_wilson_interval_brackets_the_estimate() -> None:
    lo, hi = group.wilson(90, 100)
    assert lo < 0.9 < hi
    assert group.wilson(0, 0) == (0.0, 1.0)
    assert group.wilson(100, 100)[1] == pytest.approx(1.0)


def _metrics(fgr: float, q60: float, half: float, crop: float) -> dict:
    per = {
        'T-q60': {'recall': q60},
        'T-half': {'recall': half},
        'T-crop90': {'recall': crop},
    }
    return {'false_group_rate': fgr, 'per_transform': per}


def test_bars_and_choice_rule() -> None:
    control = _metrics(0.0, 0.5, 0.5, 0.5)
    assert sweep.passes(_metrics(0.0, 0.9, 0.9, 0.8), control)
    assert not sweep.passes(_metrics(0.006, 0.99, 0.99, 0.99), control)  # false-group bar
    assert not sweep.passes(_metrics(0.0, 0.89, 0.99, 0.99), control)  # q60 bar
    assert not sweep.passes(_metrics(0.0, 0.99, 0.99, 0.79), control)  # crop bar
    assert not sweep.passes(
        _metrics(0.0, 0.99, 0.99, 0.9), _metrics(0.0, 0.5, 0.5, 0.95)
    )  # below the control
    curve = {
        0.97: _metrics(0.0, 0.95, 0.95, 0.9),
        0.96: _metrics(0.0, 0.97, 0.97, 0.95),
        0.95: _metrics(0.01, 0.99, 0.99, 0.99),
    }
    assert sweep.choose_cut(curve, control) == pytest.approx(0.97)  # highest passing
    assert sweep.choose_cut({0.99: _metrics(0.0, 0.1, 0.1, 0.1)}, control) is None


def test_split_is_seeded_disjoint_and_complete() -> None:
    a, b = sweep.split_families(10, 1)
    assert a.isdisjoint(b)
    assert a | b == set(range(10))
    assert sweep.split_families(10, 1) == (a, b)
    assert sweep.split_families(10, 2) != (a, b)


def test_row_cosine_and_hamming() -> None:
    from scripts.research.near_dup import hashes, parity

    a = _unit([[1.0, 0.0], [0.0, 1.0]])
    b = _unit([[1.0, 0.0], [1.0, 1.0]])  # cos 1 and cos 45 degrees = 0.7071
    m = parity.row_cosine(a, b)
    assert m['min'] == pytest.approx(0.70710678, abs=1e-6)
    assert m['median'] == pytest.approx((1.0 + 0.70710678) / 2, abs=1e-6)
    words = np.array(
        [[0], [0b1011], [0b1]], dtype=np.uint64
    )  # distances: 0-1 is 3 bits, 0-2 is 1, 1-2 is 2
    i, j, d = hashes.hamming_pairs(words, 2)
    assert sorted(zip(i.tolist(), j.tolist(), d.tolist(), strict=True)) == [(0, 2, 1), (1, 2, 2)]
