"""``scripts/opensearch/migrate.py`` creates and stamps what is missing and, run again, issues
no write (2F-4): ``db-migrate`` runs on every start of the stack."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.opensearch import migrate as opensearch_migrate
from src.clients.curation_opensearch.lifecycle import curation_index_names
from src.config.project_context import bind_project
from src.services.curation import index_bootstrap
from tests.opensearch.conftest import make_record
from tests.opensearch.fake_cluster import FakeCluster, FakeGlobalClient


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


async def _run(
    cluster: FakeCluster, global_client: FakeGlobalClient, records: list[ProjectRecord]
) -> opensearch_migrate.MigrateReport:
    return await opensearch_migrate.migrate(cluster, global_client, records, upgrade_pending=False)


@pytest.mark.asyncio
async def test_a_fresh_run_creates_the_global_and_the_project_indexes(
    cluster: FakeCluster, global_client: FakeGlobalClient, records: list[ProjectRecord]
) -> None:
    report = await _run(cluster, global_client, records)

    assert report.projects == ['alpha', 'beta']
    assert set(FakeGlobalClient.NAMES) <= set(cluster.indexes)
    for record in records:
        with bind_project(record):
            names = curation_index_names()
        assert set(names) <= set(cluster.indexes)
        for name in names:
            stamp = cluster.indexes[name]['_meta'][index_bootstrap.SCHEMA_META_KEY]
            assert stamp == index_bootstrap.schema_fingerprint()


@pytest.mark.asyncio
async def test_a_second_run_issues_no_write(
    cluster: FakeCluster,
    global_client: FakeGlobalClient,
    records: list[ProjectRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run(cluster, global_client, records)
    assert cluster.writes
    cluster.writes.clear()
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', set())  # a new process

    await _run(cluster, global_client, records)

    assert cluster.writes == []


@pytest.mark.asyncio
async def test_a_project_that_is_not_served_is_left_alone(
    cluster: FakeCluster, global_client: FakeGlobalClient
) -> None:
    records = [make_record('alpha'), make_record('gone', status='deleting')]

    report = await _run(cluster, global_client, records)

    assert report.projects == ['alpha']
    assert not [n for n in cluster.indexes if '__' in n and n.startswith('op_prj_gone')]


@pytest.mark.asyncio
async def test_an_index_that_cannot_be_migrated_fails_the_step(
    cluster: FakeCluster,
    global_client: FakeGlobalClient,
    records: list[ProjectRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refused(**_: object) -> None:
        msg = 'cluster_block_exception'
        raise RuntimeError(msg)

    monkeypatch.setattr(cluster.indices, 'create', refused)

    with pytest.raises(opensearch_migrate.IndexMigrationError, match='alpha'):
        await _run(cluster, global_client, records)


@pytest.mark.asyncio
async def test_each_project_graph_is_warmed_once_by_the_migrate(
    cluster: FakeCluster, global_client: FakeGlobalClient, records: list[ProjectRecord]
) -> None:
    await _run(cluster, global_client, records)

    warmups = [url for call, url in cluster.calls if call == 'transport GET' and 'warmup' in url]
    assert len(warmups) == len(records)
    assert any('alpha__items' in url for url in warmups)
