"""The capacity budget counts the shards a project really creates: one per stored index (images
and items; the lease index went with 4F-cleanup), not the six of the retired control roles."""

from __future__ import annotations

from src.clients.curation_opensearch.lifecycle import INDEX_BODIES
from src.services.projects.capacity import _build_capacity, per_project_shards


def test_a_project_costs_one_shard_per_index_it_creates() -> None:
    shards = [body['settings']['index']['number_of_shards'] for body in INDEX_BODIES.values()]

    assert per_project_shards() == sum(shards) == 2


def test_the_projected_shards_after_a_create_use_that_cost() -> None:
    capacity = _build_capacity(
        active_shards=10,
        data_nodes=1,
        max_shards_per_node=1000,
        heap_max_bytes_total=4 * 1024**3,
        extra_shards=per_project_shards(),
    )

    assert capacity.per_project_shards == 2
    assert capacity.shards_after_create == 12
