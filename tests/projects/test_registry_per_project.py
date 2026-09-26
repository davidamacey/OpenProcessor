"""P1 commit 1: the project registry (op_projects) snapshot + bootstrap."""

from __future__ import annotations

import asyncio

from src.services.projects.bootstrap import bootstrap_default_project
from src.services.projects.registry import ProjectRegistry


def test_bootstrap_is_idempotent(fake_registry_client) -> None:
    client = fake_registry_client

    async def _run() -> None:
        first = await bootstrap_default_project(client)
        second = await bootstrap_default_project(client)
        assert first.slug == second.slug == 'default'
        assert first.created_at == second.created_at  # not overwritten

    asyncio.run(_run())


def test_bootstrap_touches_no_data_index(fake_registry_client) -> None:
    """The registry doc store is the only thing written -- no reindex /
    update_by_query call shape is exercised (the fake would raise
    AttributeError if bootstrap tried one)."""
    client = fake_registry_client
    asyncio.run(bootstrap_default_project(client))
    assert set(client.docs) == {'project:default', 'meta:projects_revision'}


def test_registry_snapshot_refreshes_on_revision_change(fake_registry_client) -> None:
    client = fake_registry_client

    async def _run() -> None:
        await bootstrap_default_project(client)
        registry = ProjectRegistry(lambda: client)
        await registry.ensure_fresh()
        assert set(registry.snapshot()) == {'default'}

        # A second project appears only after ensure_fresh re-reads.
        from datetime import UTC, datetime

        from src.config.curation import base_curation_config
        from src.config.projects import ProjectRecord, resources_for_new
        from src.services.projects.registry import record_to_doc

        now = datetime.now(UTC).isoformat()
        record = ProjectRecord(
            slug='alpha',
            display_name='Alpha',
            description='',
            status='active',
            revision=1,
            created_at=now,
            updated_at=now,
            origin=None,
            resources=resources_for_new('alpha', base_curation_config()),
        )
        await client.index(index='op_projects', id='project:alpha', body=record_to_doc(record))
        await client.index(index='op_projects', id='meta:projects_revision', body={'revision': 2})

        await registry.ensure_fresh()
        assert set(registry.snapshot()) == {'default', 'alpha'}
        alpha = registry.get('alpha')
        assert alpha is not None
        assert alpha.resources.model_prefix == 'alpha__'

    asyncio.run(_run())


def test_registry_skips_search_when_revision_unchanged(fake_registry_client) -> None:
    """Only a GET happens when the counter has not moved -- no _search."""
    client = fake_registry_client
    search_calls = []
    orig_search = client.search

    async def _spy_search(**kwargs):
        search_calls.append(kwargs)
        return await orig_search(**kwargs)

    client.search = _spy_search  # type: ignore[method-assign]

    async def _run() -> None:
        await bootstrap_default_project(client)
        registry = ProjectRegistry(lambda: client)
        await registry.ensure_fresh()
        assert len(search_calls) == 1
        await registry.ensure_fresh()
        assert len(search_calls) == 1  # unchanged revision -> no second search

    asyncio.run(_run())
