"""Tests for :mod:`src.services.detection.frame_dedup`: the pure-numpy union-find grouping."""

from __future__ import annotations

import numpy as np

from src.services.detection.frame_dedup import most_central, near_dup_groups


def _unit(vecs: list[list[float]]) -> np.ndarray:
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / norms).astype(np.float32)


class TestNearDupGroups:
    def test_fewer_than_two_rows_returns_empty(self) -> None:
        assert near_dup_groups(_unit([[1.0, 0.0]])) == []

    def test_identical_vectors_group_together(self) -> None:
        mat = _unit([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        groups = near_dup_groups(mat, threshold=0.98)
        assert groups == [[0, 1]]

    def test_orthogonal_vectors_do_not_group(self) -> None:
        mat = _unit([[1.0, 0.0], [0.0, 1.0]])
        assert near_dup_groups(mat, threshold=0.98) == []

    def test_transitive_grouping_via_chain(self) -> None:
        # a~b (cos ~0.999), b~c (cos ~0.999), a and c both above threshold
        # transitively through b.
        mat = _unit([[1.0, 0.0], [0.999, 0.0447], [0.996, 0.0894]])
        groups = near_dup_groups(mat, threshold=0.98)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}


class TestMostCentral:
    def test_returns_index_of_most_representative_member(self) -> None:
        # Third vector is the average direction of the first two -> most central.
        mat = _unit([[1.0, 0.1], [0.1, 1.0], [1.0, 1.0]])
        assert most_central(mat, [0, 1, 2]) == 2
