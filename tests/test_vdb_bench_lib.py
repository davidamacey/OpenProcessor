"""Pure logic of the vector store benchmark (scripts/bench/vectordb/vdb_lib.py)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.bench.vectordb import vdb_lib as lib


def _unit(rng: np.random.Generator, n: int, d: int = 16) -> np.ndarray:
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_exact_topk_matches_bruteforce_and_respects_mask() -> None:
    rng = np.random.default_rng(1)
    vectors, queries = _unit(rng, 500), _unit(rng, 7)
    mask = rng.random(500) < 0.3
    got = lib.exact_topk(vectors, queries, mask, 10)
    for qi, q in enumerate(queries):
        scores = np.where(mask, vectors @ q, -np.inf)
        assert list(got[qi]) == list(np.argsort(-scores)[:10])
        assert mask[got[qi]].all()


def test_exact_topk_pads_when_fewer_matches_than_k() -> None:
    rng = np.random.default_rng(2)
    vectors, queries = _unit(rng, 50), _unit(rng, 2)
    mask = np.zeros(50, dtype=bool)
    mask[[3, 9]] = True
    got = lib.exact_topk(vectors, queries, mask, 5)
    assert set(got[0][:2]) == {3, 9}
    assert (got[0][2:] == -1).all()
    empty = lib.exact_topk(vectors, queries, np.zeros(50, dtype=bool), 5)
    assert (empty == -1).all()


def test_recall_counts_overlap_and_ignores_padding() -> None:
    truth = np.array([[1, 2, 3, 4], [7, 8, -1, -1]])
    found = [[1, 2, 9, 9], [8]]
    assert lib.recall_at_k(truth, found, 4) == pytest.approx((0.5 + 0.5) / 2)
    assert lib.recall_at_k(truth, found, 2) == pytest.approx((1.0 + 0.5) / 2)


def test_recall_skips_queries_with_no_exact_answer() -> None:
    truth = np.array([[-1, -1], [5, 6]])
    assert lib.recall_at_k(truth, [[], [5, 6]], 2) == 1.0
    assert np.isnan(lib.recall_at_k(truth[:1], [[]], 2))


def test_short_rate_flags_fewer_results_than_the_exact_answer_has() -> None:
    truth = np.array([[1, 2, 3], [4, 5, -1]])
    assert lib.short_rate([[1, 2], [4, 5]], truth, 3) == 0.5


def test_percentile_is_nearest_rank() -> None:
    values = [float(i) for i in range(1, 101)]
    assert lib.percentile(values, 50) == 50.0
    assert lib.percentile(values, 95) == 95.0
    assert lib.percentile(values, 99) == 99.0
    assert lib.percentile([3.0], 99) == 3.0
    assert np.isnan(lib.percentile([], 50))


def test_filter_mask_and_selectivity_fields() -> None:
    image_ids = np.repeat(np.arange(2000), 5)
    fields = lib.make_filter_fields(image_ids, np.arange(len(image_ids)), seed=3)
    fields['cluster_id'] = (np.arange(len(image_ids)) % 4).astype(np.int64)
    fields['class_name'] = np.where(np.arange(len(image_ids)) % 2 == 0, 'car', 'dog')
    n = len(image_ids)
    flts = {f.name: f for f in lib.filter_matrix(0, 'car', 1)}
    assert lib.filter_mask(flts['none'], fields).all()
    assert abs(lib.filter_mask(flts['project'], fields).mean() - 0.10) < 0.03
    assert abs(lib.filter_mask(flts['sel10'], fields).mean() - 0.10) < 0.02
    both = lib.filter_mask(flts['project_class'], fields)
    assert (fields['class_name'][both] == 'car').all()
    assert (fields['project'][both] == 0).all()
    review = lib.filter_mask(flts['review_default'], fields)
    for flag in ('class_validated', 'test_holdout', 'review_dismissed'):
        assert not fields[flag][review].any()
    assert review.sum() < lib.filter_mask(flts['project'], fields).sum() <= n
    # Crops of one image share a project: the project is a function of the image only.
    by_image = fields['project'].reshape(2000, 5)
    assert (by_image == by_image[:, :1]).all()
