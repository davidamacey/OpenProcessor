"""``op.periodic.cluster_refresh``: starts ``cluster_run`` (never ``autolabel``) for a project when
the item pool grew past the profile's growth rule and ingest is quiet, and only when a ``cluster``
consumer is live (a fresh ``worker_capabilities`` row) and no assist loop holds the clusters.

The schedule's own rules (deferral, backoff, coalescing) are pinned in
``tests/curation/test_cluster_refresh_schedule.py``; this file runs the tick on a real database.
"""

from __future__ import annotations

from typing import Any

import pytest
from fixtures.job_rows import make_project, sql

from src.core.exceptions import RequestRejectedError
from src.domain.embedding_spaces import Target
from src.services.curation.assist_loop import service as assist_service
from src.services.periodic import cluster_refresh
from src.services.periodic.runner import run_periodic


pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures('pg_registry', 'job_env', 'producer', 'wakes'),
]

DEFER = 1800.0


class FakeOpenSearch:
    def __init__(self) -> None:
        self.pool = 1000
        self.unfinished = 0
        self.recent_ingest = False
        self.closed = False

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, int]:
        assert index == 'items'
        if 'terms' in body['query']:
            return {'count': self.unfinished}
        return {'count': self.pool}

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        assert index == 'images'
        assert body['size'] == 0
        return {'hits': {'total': {'value': 1 if self.recent_ingest else 0}}}

    async def close(self) -> None:
        self.closed = True


class FakeSpace:
    vector_field = 'pe_embedding'


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, pg) -> FakeOpenSearch:
    sql(pg, 'DELETE FROM worker_capabilities')
    client = FakeOpenSearch()

    async def opensearch() -> FakeOpenSearch:
        return client

    async def space(target: Target) -> FakeSpace:
        assert target is Target.ITEM
        return FakeSpace()

    monkeypatch.setattr(cluster_refresh, 'opensearch_client', opensearch)
    monkeypatch.setattr(cluster_refresh, 'embedding_space', space)
    monkeypatch.setattr(cluster_refresh, 'items_index', lambda: 'items')
    monkeypatch.setattr(cluster_refresh, 'images_index', lambda: 'images')
    return client


def consumer(pg, *, age: str = '0 seconds') -> None:
    sql(
        pg,
        'INSERT INTO worker_capabilities (worker_id, queue, host, capabilities, seen_at) '
        "VALUES ('cluster@h', 'cluster', 'h', '{}', now() - %s::interval) "
        'ON CONFLICT (worker_id) DO UPDATE SET seen_at = excluded.seen_at',
        age,
    )


def built(pg, project, n_items: int, status: str = 'ready') -> None:
    sql(
        pg,
        'INSERT INTO cluster_partitions (project_id, target, level, embedding_field, '
        'embedding_model, preprocess_id, method, method_version, params, pool_filter, '
        'input_digest, seed, backend, backend_detail, status, n_items, created_by) VALUES '
        "(%s, 'item', 'class', 'pe_embedding', 'pe@v1', 'p', 'ivf', '1', '{}', '{}', 'd', 1, "
        "'cpu', '{}', %s, %s, 'test')",
        project,
        status,
        n_items,
    )


def runs(pg, project) -> list[tuple[Any, ...]]:
    return sql(
        pg,
        "SELECT status, queue, requested_by, params FROM jobs WHERE kind = 'cluster_run' "
        'AND project_id = %s',
        project,
    )


def tick(now: float = 10_000.0) -> Any:
    return run_periodic('cluster_refresh', cluster_refresh.tick(now=now))


def test_growth_past_the_threshold_with_a_live_consumer_starts_one_cluster_run(pg, world) -> None:
    project = make_project(pg)
    consumer(pg)
    tick()
    tick(10_060.0)  # the live run keeps the second tick from starting another
    [(status, queue, by, params)] = runs(pg, project)
    assert (status, queue, by) == ('queued', 'cluster', 'beat')
    assert params['target'] == 'item'
    assert params['scope_key'] == 'item:class:pe_embedding'
    assert world.closed
    assert sql(pg, "SELECT count(*) FROM jobs WHERE kind = 'autolabel'")[0][0] == 0


def test_no_live_cluster_consumer_starts_nothing(pg, world) -> None:
    project = make_project(pg)
    tick()
    assert runs(pg, project) == []
    consumer(pg, age='10 minutes')  # a row that stopped refreshing is not a consumer
    tick()
    assert runs(pg, project) == []


def test_growth_below_the_threshold_of_the_newest_partition_starts_nothing(pg, world) -> None:
    project = make_project(pg)
    consumer(pg)
    built(pg, project, 900)  # default growth threshold is 200; the pool is 1000
    tick()
    assert runs(pg, project) == []
    world.pool = 1100
    tick()  # the pool moved since the last tick: not quiet yet
    assert runs(pg, project) == []
    tick()
    assert len(runs(pg, project)) == 1


def test_recent_ingest_defers_until_the_deferral_ends(pg, world) -> None:
    project = make_project(pg)
    consumer(pg)
    world.recent_ingest = True
    tick(10_000.0)
    assert runs(pg, project) == []
    tick(10_000.0 + DEFER - 1)
    assert runs(pg, project) == []
    tick(10_000.0 + DEFER)  # the clock the first tick stored carries over to later ticks
    assert len(runs(pg, project)) == 1


def test_a_live_assist_loop_holds_the_refresh(pg, world, monkeypatch) -> None:
    project = make_project(pg)
    consumer(pg)

    async def held(_session: Any) -> None:
        raise RequestRejectedError(409, {'error': 'assist_loop_running'})

    monkeypatch.setattr(assist_service, 'require_no_live_loop', held)
    tick()
    assert runs(pg, project) == []


def test_a_failed_run_is_retried_only_after_the_backoff(pg, world) -> None:
    project = make_project(pg)
    consumer(pg)
    tick(10_000.0)
    sql(pg, "UPDATE jobs SET status = 'failed', finished_at = now() WHERE kind = 'cluster_run'")
    tick(10_100.0)  # notices the failure, waits the profile's 600 s
    tick(10_100.0 + 599)
    assert len(runs(pg, project)) == 1
    tick(10_100.0 + 600)
    assert len(runs(pg, project)) == 2


def test_a_project_that_turned_automatic_clustering_off_is_left_alone(pg, world) -> None:
    project = make_project(pg)
    consumer(pg)
    sql(
        pg,
        'INSERT INTO project_settings (project_id, ingest_policy, updated_by) '
        "VALUES (%s, %s, 'test')",
        project,
        '{"clustering": {"automatic": false}}',
    )
    tick()
    assert runs(pg, project) == []


def test_a_clock_carried_over_from_060_names_no_job_and_costs_one_backoff(pg, world) -> None:
    project = make_project(pg)
    consumer(pg)
    sql(
        pg,
        "INSERT INTO project_state (project_id, key, value) VALUES (%s, 'cluster_refresh', %s)",
        project,
        '{"last_trained_count": 120, "job_id": "al_20260101_1"}',
    )
    tick(10_000.0)
    assert runs(pg, project) == []
    tick(10_000.0 + 600)
    assert len(runs(pg, project)) == 1
