"""An assist loop that names no ``scope.min_root_size`` sizes it from the clusters it will work on:
the usual 40, lowered to the size that still leaves enough roots to be worth a run."""

from __future__ import annotations

import pytest

from src.schemas.curation.assist_loop import LoopScope
from src.services.curation.assist_loop.prepare import sized_min_root_size


def test_a_partition_of_big_clusters_keeps_the_usual_threshold() -> None:
    assert sized_min_root_size([80, 70, 60, 55, 50, 48, 45, 44, 43, 42, 41, 5]) == 40


def test_a_partition_of_small_clusters_lowers_it_to_keep_ten_roots() -> None:
    # The walk's data: 203 clusters, the largest 24.
    sizes = [24, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, *([3] * 191)]
    assert sized_min_root_size(sizes) == 14


def test_a_partition_with_few_clusters_takes_them_all() -> None:
    assert sized_min_root_size([6, 3, 2]) == 2


def test_no_clusters_leaves_the_usual_threshold_for_the_no_roots_refusal() -> None:
    assert sized_min_root_size([]) == 40


def test_the_scope_leaves_the_threshold_to_the_server_unless_given() -> None:
    assert LoopScope().min_root_size is None
    assert LoopScope(min_root_size=12).min_root_size == 12
    with pytest.raises(ValueError, match='greater than or equal to 1'):
        LoopScope(min_root_size=0)
