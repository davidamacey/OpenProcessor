"""Pure logic of the cluster ladder (scripts/bench/cluster_ladder.py)."""

from __future__ import annotations

from scripts.bench import cluster_ladder as ladder


def test_whole_pass_counts_the_stages_that_follow_the_pass() -> None:
    row = {'wall_s': 100.0, 'geometry_s': 30.0, 'assign_only_s': 20.0}
    assert ladder.whole_pass_ms_per_crop(row, 100_000) == 1.5


def test_whole_pass_without_the_optional_stages_is_the_pass_alone() -> None:
    assert ladder.whole_pass_ms_per_crop({'wall_s': 250.0}, 100_000) == 2.5
