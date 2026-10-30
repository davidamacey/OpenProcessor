"""The loop recursion and the hold-back guard (Z4): every expected value is worked out beside it.

Gate numbers are the loop's published tables (theta 0.9, alpha_stage 0.025): a sample of 30 for a
cluster of 150, 23 for 50 members.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.cluster_diag import gate_sim as gs, loop_sim as ls


def _vectors(n: int) -> np.ndarray:
    v = np.zeros((n, 3), dtype=np.float32)
    v[:, 0] = 1.0
    return v


def _no_split(_members: np.ndarray, _path: str) -> list[np.ndarray]:
    return []


def _must_not_split(_members: np.ndarray, _path: str) -> list[np.ndarray]:
    raise AssertionError('this node must not be split')


def _by_parity(members: np.ndarray, _path: str) -> list[np.ndarray]:
    """Two children that separate the 'a' (even) and 'b' (odd) members."""
    return [members[members % 2 == 0], members[members % 2 == 1]]


def _by_class(members: np.ndarray, _path: str) -> list[np.ndarray]:
    """Children that separate the first 400 indices from the rest."""
    kids = [members[members < 400], members[members >= 400]]
    return [k for k in kids if len(k)]


def _halves(members: np.ndarray, _path: str) -> list[np.ndarray]:
    """Two children that keep the class mixture (the first and second half by position)."""
    mid = len(members) // 2
    return [members[:mid], members[mid:]]


def _loop(
    truth: list[str], splitter: ls.Splitter, **kw: object
) -> tuple[ls.Run, dict[str, object]]:
    codes = np.zeros(len(truth), dtype=np.int64)
    seed = int(kw.pop('seed', 0))  # type: ignore[call-overload]
    run = ls.run_loop(codes, truth, _vectors(len(truth)), splitter, seed=seed, **kw)  # type: ignore[arg-type]
    return run, ls.measure(run, truth)


def test_a_pure_cluster_passes_on_the_first_sample() -> None:
    run, m = _loop(['car'] * 150, _must_not_split)
    assert (len(run.asked), len(run.propagated), len(run.review)) == (30, 120, 0)
    assert (m['u1'], m['u2'], m['u3']) == (0.2, 1.0, 0.8)
    assert run.nodes == {'pass': 1}


def test_a_tiny_pure_cluster_is_asked_whole_and_passes() -> None:
    run, m = _loop(['car'] * 5, _must_not_split)
    assert (len(run.asked), len(run.propagated)) == (5, 0)
    assert m['u1'] == 1.0


def test_a_mixed_cluster_splits_and_the_children_reuse_the_parent_answers() -> None:
    truth = ['a', 'b'] * 50
    run, m = _loop(truth, _by_parity)
    # root: 30 asked, about half dissent, fails. Each pure child of 50 needs 22 in total, and the
    # parent's answers that fall in it count toward them, so the whole run asks 2 * 22 = 44, not
    # 30 + 44
    assert run.nodes == {'split': 1, 'pass': 2}
    assert len(run.asked) == 2 * gs.required_sample(50, 0.9, 0.025, 0) == 44
    assert len(run.propagated) == 56
    assert (m['u1'], m['u2']) == (0.44, 1.0)


def test_a_cluster_that_cannot_be_split_goes_to_review_whole() -> None:
    run, m = _loop(['a', 'b'] * 50, _no_split)
    assert run.nodes == {'review_leaf': 1}
    assert len(run.asked) + len(run.review) == 100
    assert (m['u1'], m['propagated']) == (1.0, 0)


def test_a_small_failing_node_is_reviewed_without_a_split() -> None:
    run, m = _loop(['a', 'b'] * 15, _must_not_split)
    assert run.nodes == {'review_small': 1}
    assert m['u1'] == 1.0


def test_the_recursion_stops_at_the_maximum_depth() -> None:
    # 400 -> 200 -> 100 -> 50: the 50-member nodes sit at depth 3 and are reviewed even though
    # they are not under the minimum size
    run, m = _loop(['a', 'b'] * 200, _halves)
    assert run.nodes == {'split': 7, 'review_leaf': 8}
    assert m['u1'] == 1.0


def test_one_dissent_in_the_first_sample_tops_the_sample_up() -> None:
    truth = ['car'] * 149 + ['dog']
    seen = set()
    for seed in range(60):
        run, _ = _loop(truth, _must_not_split, seed=seed)
        dog_asked = 149 in run.asked
        # first sample 30; a dissent among them needs 44 in total; otherwise the dog takes the label
        assert len(run.asked) == (44 if dog_asked else 30)
        if not dog_asked:
            assert run.propagated[149] == 'car'
        seen.add(dog_asked)
    assert seen == {True, False}


def test_two_dissents_fail_the_node() -> None:
    truth = ['car'] * 148 + ['dog', 'dog']
    seen = set()
    for seed in range(400):
        run, _ = _loop(truth, _no_split, seed=seed)
        both = sum(i in run.asked for i in (148, 149)) == 2
        if both:
            # both in the first 30: two dissents fail at once, with no top-up (30 asked). One in
            # the first 30 and the other found by the top-up: 44 asked, still short of the 56
            # that two dissents need, so no pass either
            assert run.nodes == {'review_leaf': 1}
            assert len(run.asked) in (30, 44)
            seen.add(len(run.asked))
    assert seen == {30, 44}


def test_a_top_class_outside_the_registry_cannot_pass() -> None:
    run, _ = _loop(['fox'] * 150, _no_split, registry=frozenset({'dog'}))
    assert run.nodes == {'review_leaf': 1}
    assert not run.propagated
    run, _ = _loop(['fox'] * 150, _no_split, registry=frozenset({'fox'}))
    assert run.nodes == {'pass': 1}


def test_held_back_members_are_the_ones_beyond_the_sample_percentile() -> None:
    angle = np.concatenate(
        [
            np.linspace(-0.02, 0.02, 20),  # the sample: its own spread sets the limit
            [0.0, 0.01, -0.015],  # members inside that spread
            [np.pi / 2] * 3,  # three outliers, 90 degrees away
        ]
    )
    v = np.stack([np.cos(angle), np.sin(angle)], axis=1).astype(np.float32)
    got = ls.held_back(v, np.arange(26), np.arange(20), 95.0)
    assert got.tolist() == [23, 24, 25]  # sampled members never count; inliers stay
    # a member just past the sample's widest spread is held back too: the limit is the sample's
    v[20] = [np.cos(0.05), np.sin(0.05)]
    assert ls.held_back(v, np.arange(26), np.arange(20), 95.0).tolist() == [20, 23, 24, 25]


def test_the_guard_turns_far_members_into_reviews_and_nothing_else() -> None:
    rng = np.random.default_rng(3)
    n = 200
    v = np.zeros((n, 2), dtype=np.float32)
    angle = rng.normal(0.0, 0.02, n)
    angle[:10] = np.pi / 2  # ten far members of the same class
    v[:, 0], v[:, 1] = np.cos(angle), np.sin(angle)
    truth = ['car'] * n
    codes = np.zeros(n, dtype=np.int64)
    plain = ls.run_loop(codes, truth, v, _no_split, seed=4)
    guarded = ls.run_loop(codes, truth, v, _no_split, seed=4, guard=True)
    assert set(guarded.asked) == set(plain.asked)
    assert set(guarded.propagated) | set(guarded.held) == set(plain.propagated)
    assert not set(guarded.propagated) & set(guarded.held)
    assert guarded.review == set(guarded.held)
    assert guarded.held, 'the far members should be held back'
    # the limit is the sample's 95th percentile, so ordinary members are held back only rarely
    assert len(guarded.held) <= 10 + 0.2 * (n - 10)
    far_unasked = set(range(10)) - set(guarded.asked)
    assert far_unasked <= set(guarded.held), 'every far member that was not asked is held back'
    m = ls.measure(guarded, truth)
    assert m['decisions'] == len(guarded.asked) + len(guarded.held)
    assert m['held_back'] == len(guarded.held)
    assert m['held_back_would_be_wrong'] == 0  # same class: the guard only cost effort here


def test_measure_counts_wrong_and_open_set_propagations() -> None:
    truth = ['dog', 'dog', 'dog', 'fox', 'fox', 'cat']
    run = ls.Run(
        asked={0: 'dog', 5: 'cat'},
        review={4},
        propagated={1: 'dog', 2: 'dog', 3: 'dog'},
        held={4: 'dog'},
    )
    m = ls.measure(run, truth, open_set={'fox'})
    assert m['decisions'] == 3
    assert m['u1'] == 0.5
    assert (m['propagated'], m['wrong_propagated']) == (3, 1)
    assert m['u2'] == pytest.approx(2 / 3)
    assert (m['open_propagated'], m['open_wrong'], m['u2_open']) == (1, 1, 0.0)
    assert (m['held_back'], m['held_back_would_be_wrong'], m['held_back_open_set']) == (1, 1, 1)


def test_an_item_decided_twice_is_refused() -> None:
    run = ls.Run(asked={0: 'a'}, review={0})
    with pytest.raises(ValueError, match='decided twice'):
        ls.measure(run, ['a'])


def test_a_noisy_labeler_is_wrong_and_its_answers_are_reused() -> None:
    truth = ['car'] * 400 + ['dog'] * 400
    codes = np.zeros(800, dtype=np.int64)
    run = ls.run_loop(codes, truth, _vectors(800), _by_class, seed=1, noise=0.5)
    wrong = sum(1 for i, a in run.asked.items() if a != truth[i])
    assert wrong > 0
    # an answer is memoised per item, so asking again in a child cannot change it
    again = ls.run_loop(codes, truth, _vectors(800), _by_class, seed=1, noise=0.5)
    assert again.asked == run.asked


def test_seeded_loop_reports_mean_min_max_and_per_seed_values() -> None:
    truth = ['car'] * 150
    codes = np.zeros(150, dtype=np.int64)
    out = ls.seeded_loop(codes, truth, _vectors(150), _no_split, seeds=[0, 1, 2])
    assert out['u1'] == pytest.approx({'mean': 0.2, 'min': 0.2, 'max': 0.2})
    assert out['per_seed']['asked'] == [30, 30, 30]
    assert out['u2_open'] is None  # no open-set item exists


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    centres = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    rng = np.random.default_rng(0)
    pts = np.concatenate([c + rng.normal(0, 0.02, (10, 4)) for c in centres]).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts, np.repeat(np.arange(4), 10)


def test_the_ahc_splitter_finds_four_groups_ordered_by_smallest_member() -> None:
    pts, blob = _blobs()
    order = np.random.default_rng(1).permutation(40)
    shuffled, shuffled_blob = pts[order], blob[order]
    kids = ls.AhcSplitter(shuffled)(np.arange(40), 'r0')
    assert len(kids) == 4
    assert [int(k[0]) for k in kids] == sorted(int(k[0]) for k in kids)
    for k in kids:
        assert len(set(shuffled_blob[k].tolist())) == 1
        assert len(k) == 10


def test_the_splitter_refuses_too_few_members_and_uses_kmeans_above_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pts, blob = _blobs()
    sp = ls.AhcSplitter(pts)
    assert sp(np.arange(3), 'r0') == []
    monkeypatch.setattr(ls, 'MAX_REFINE', 20)
    kids = sp(np.arange(40), 'r0')
    assert len(kids) == 4
    assert all(len(set(blob[k].tolist())) == 1 for k in kids)
    assert [int(k[0]) for k in kids] == sorted(int(k[0]) for k in kids)
    assert not sp._cache, 'the k-means path is not cached'


def test_path_seed_is_stable_and_depends_on_both_parts() -> None:
    assert ls.path_seed(1, 'r0.1') == ls.path_seed(1, 'r0.1')
    assert len({ls.path_seed(1, 'r0.1'), ls.path_seed(2, 'r0.1'), ls.path_seed(1, 'r0.2')}) == 3


def test_the_registered_loop_defaults_and_guard_percentile() -> None:
    assert (ls.BRANCHING, ls.MAX_DEPTH, ls.MIN_NODE, ls.MAX_REFINE) == (4, 3, 40, 8000)
    assert ls.GUARD_PERCENTILE == 95.0
