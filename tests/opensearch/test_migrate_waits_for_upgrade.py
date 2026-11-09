"""While the 0.6.0 control data is not loaded the projects are not in PostgreSQL, so the migrate
touches only the global indexes; after the load the project indexes are created and stamped
(02c section 1.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.opensearch import migrate as opensearch_migrate
from src.services.curation import index_bootstrap
from tests.opensearch.fake_cluster import FakeCluster, FakeGlobalClient


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


@pytest.mark.asyncio
async def test_with_the_upgrade_pending_only_the_global_indexes_are_touched(
    cluster: FakeCluster, global_client: FakeGlobalClient, records: list[ProjectRecord]
) -> None:
    report = await opensearch_migrate.migrate(cluster, global_client, records, upgrade_pending=True)

    assert report.skipped_for_upgrade is True
    assert report.projects == []
    assert set(cluster.indexes) == set(FakeGlobalClient.NAMES)
    assert {name for _, name in cluster.writes} == set(FakeGlobalClient.NAMES)


@pytest.mark.asyncio
async def test_after_the_load_the_project_indexes_are_stamped(
    cluster: FakeCluster, global_client: FakeGlobalClient, records: list[ProjectRecord]
) -> None:
    await opensearch_migrate.migrate(cluster, global_client, [], upgrade_pending=True)

    report = await opensearch_migrate.migrate(
        cluster, global_client, records, upgrade_pending=False
    )

    assert report.projects == ['alpha', 'beta']
    stamped = [
        name
        for name, idx in cluster.indexes.items()
        if idx['_meta'].get(index_bootstrap.SCHEMA_META_KEY) == index_bootstrap.schema_fingerprint()
    ]
    assert len(stamped) == 4  # images and items of two projects


def test_the_pending_check_reads_the_state_and_the_legacy_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Conn:
        def __enter__(self) -> Conn:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(opensearch_migrate, 'one_off_connection', lambda *_a, **_k: Conn())
    for state, legacy, expected in [
        ('in_progress', False, True),
        ('completed', True, True),
        ('completed', False, False),
        (None, False, False),
    ]:
        monkeypatch.setattr(opensearch_migrate, 'upgrade_state', lambda _c, s=state: s)
        monkeypatch.setattr(
            opensearch_migrate, 'legacy_projects_index_exists', lambda lg=legacy: lg
        )

        assert opensearch_migrate.upgrade_pending() is expected
