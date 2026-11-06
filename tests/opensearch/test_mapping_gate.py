"""The API refuses to start when an index of an active project is missing or carries another
schema stamp, naming the index (2F-4, ``opensearch_mapping_not_current``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from scripts.opensearch import migrate as opensearch_migrate
from src.clients.curation_opensearch.lifecycle import curation_index_names
from src.config.project_context import bind_project
from src.core.dependencies import (
    OpenSearchClientFactory,
    bootstrap_opensearch_indexes as startup_check,
)
from src.services.curation import index_bootstrap
from src.services.curation.index_bootstrap import MappingNotCurrentError, check_mappings_current


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from tests.opensearch.fake_cluster import FakeCluster, FakeGlobalClient


@pytest_asyncio.fixture
async def migrated(
    cluster: FakeCluster, global_client: FakeGlobalClient, records: list[ProjectRecord]
) -> list[str]:
    await opensearch_migrate.migrate(cluster, global_client, records, upgrade_pending=False)
    with bind_project(records[0]):
        return curation_index_names()


@pytest.mark.asyncio
async def test_current_stamps_pass_in_one_mapping_call(
    cluster: FakeCluster, migrated: list[str]
) -> None:
    cluster.calls.clear()

    await check_mappings_current(cluster, migrated)

    assert [c for c, _ in cluster.calls] == ['indices.get_mapping']


@pytest.mark.asyncio
async def test_a_stale_stamp_refuses_the_start_naming_the_index(
    cluster: FakeCluster, migrated: list[str]
) -> None:
    stale = migrated[1]
    cluster.indexes[stale]['_meta'][index_bootstrap.SCHEMA_META_KEY] = 'an-older-schema'

    with pytest.raises(MappingNotCurrentError, match='opensearch_mapping_not_current') as err:
        await check_mappings_current(cluster, migrated)

    assert err.value.indexes == [stale]
    assert stale in str(err.value)


@pytest.mark.asyncio
async def test_a_missing_index_and_an_unstamped_one_are_both_named(
    cluster: FakeCluster, migrated: list[str]
) -> None:
    missing, unstamped = migrated[0], migrated[1]
    del cluster.indexes[missing]
    cluster.indexes[unstamped]['_meta'] = {}

    with pytest.raises(MappingNotCurrentError) as err:
        await check_mappings_current(cluster, migrated)

    assert err.value.indexes == [missing, unstamped]


@pytest.mark.asyncio
async def test_the_startup_check_binds_each_active_project_and_refuses_a_stale_one(
    cluster: FakeCluster,
    migrated: list[str],
    records: list[ProjectRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from src.config.projects import ProjectRecord as Record

    class Wrapper:
        client = cluster

    async def get_client() -> Wrapper:
        return Wrapper()

    snapshot = {r.slug: r for r in records}
    snapshot['gone'] = Record(**{**records[0].__dict__, 'slug': 'gone', 'status': 'deleted'})
    monkeypatch.setattr(OpenSearchClientFactory, 'get_client', get_client)
    monkeypatch.setattr(
        'src.services.projects.registry.get_project_registry',
        lambda: SimpleNamespace(snapshot=lambda: snapshot),
    )

    assert await startup_check() is None  # deleted 'gone' is skipped

    stale = next(n for n in cluster.indexes if n.startswith('op_prj_beta__items'))
    cluster.indexes[stale]['_meta'] = {}
    with pytest.raises(MappingNotCurrentError, match=stale):
        await startup_check()
