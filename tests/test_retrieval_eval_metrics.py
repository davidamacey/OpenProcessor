"""Hand-computed checks for the retrieval baseline metrics (scripts/bench/retrieval_eval_metrics.py)."""

from __future__ import annotations

import math

from scripts.bench import retrieval_eval_metrics as m


def test_rank_of_is_one_based_and_none_when_absent() -> None:
    assert m.rank_of(['a', 'b', 'c'], {'c'}) == 3
    assert m.rank_of(['a', 'b', 'c'], {'b', 'c'}) == 2
    assert m.rank_of(['a'], {'z'}) is None


def test_single_relevant_metrics_hand_computed() -> None:
    # ranks 1, 3, 10, 11 (beyond the cutoff), absent
    got = m.single_relevant_metrics([1, 3, 10, 11, None])
    assert got['recall@1'] == 1 / 5
    assert got['recall@5'] == 2 / 5
    assert got['recall@10'] == 3 / 5
    assert math.isclose(got['mrr@10'], (1 + 1 / 3 + 1 / 10) / 5)
    assert math.isclose(got['ndcg@10'], (1 + 1 / math.log2(4) + 1 / math.log2(11)) / 5)


def test_perfect_and_empty_rankings() -> None:
    assert m.single_relevant_metrics([1, 1])['mrr@10'] == 1.0
    assert m.single_relevant_metrics([None, None])['recall@10'] == 0.0


def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    ranks: list[int | None] = [1] * 60 + [4] * 20 + [None] * 20
    lo, hi = m.bootstrap_ci(ranks, 'recall@1')
    assert lo <= 0.6 <= hi
    assert hi - lo < 0.25


def test_percentile_nearest_rank() -> None:
    vals = [float(i) for i in range(1, 101)]
    assert m.pct(vals, 0.5) == 50.0
    assert m.pct(vals, 0.95) == 95.0
    assert m.pct(vals, 0.99) == 99.0


def test_jaccard() -> None:
    assert m.jaccard({'a', 'b'}, {'b', 'c'}) == 1 / 3
    assert m.jaccard({'a'}, {'a'}) == 1.0
    assert m.jaccard(set(), set()) == 0.0
