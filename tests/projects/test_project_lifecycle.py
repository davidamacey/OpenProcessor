"""P3: create / patch / archive / unarchive (§4, §10 P3)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch, seed_default_project


@pytest.fixture(autouse=True)
def _reset_registry(tmp_path, monkeypatch):
    import src.config.curation as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)
    yield
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


def _registry_for(client: FakeLifecycleOpenSearch) -> ProjectRegistry:
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)
    return registry


@pytest.fixture(autouse=True)
def _noop_ensure_indexes():
    """P3's scope is lifecycle logic, not the mapping/migration
    machinery P1 already owns and tests -- stub the (heavy, already
    covered elsewhere) index bootstrap so these tests exercise only
    lifecycle.py's own decisions."""
    with patch('src.routers.curation._common._ensure_indexes', new=AsyncMock()):
        yield


def test_create_project_slug_invalid() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.create_project(client, slug='Bad_Slug', display_name='x'))
    assert exc_info.value.detail['error'] == 'slug_invalid'


def test_create_project_success() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    record, warnings = asyncio.run(
        lifecycle.create_project(client, slug='cars', display_name='Cars')
    )
    assert record.status == 'active'
    assert record.slug == 'cars'
    assert warnings == []


def test_create_project_slug_taken() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='cars', display_name='Cars'))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.create_project(client, slug='cars', display_name='Cars 2'))
    assert exc_info.value.detail['error'] == 'slug_taken'


def test_create_project_concurrent_same_slug_exactly_one_wins() -> None:
    """M1: two concurrent creates of the same slug must not both land as
    'active'. The storage-level op_type='create' guard (not just the
    in-memory snapshot check) must decide the race."""
    client = FakeLifecycleOpenSearch()
    _registry_for(client)

    async def _race() -> tuple[Any, ...]:
        return await asyncio.gather(
            lifecycle.create_project(client, slug='zeta', display_name='First'),
            lifecycle.create_project(client, slug='zeta', display_name='Second'),
            return_exceptions=True,
        )

    results = asyncio.run(_race())
    successes = [r for r in results if not isinstance(r, BaseException)]
    failures = [r for r in results if isinstance(r, BaseException)]
    assert len(successes) == 1, results
    assert len(failures) == 1, results
    assert isinstance(failures[0], HTTPException)
    assert failures[0].detail['error'] == 'slug_taken'


def test_write_record_create_op_type_refuses_second_writer() -> None:
    """M1's storage-level primitive, isolated from create_project's own
    in-memory snapshot check (which alone cannot decide a real race --
    see the module docstring): two writes of the *same* building record
    with ``op_type='create'`` for one slug must let exactly one through,
    even though both pass an identical in-memory precondition check."""
    from src.config.curation import base_curation_config
    from src.config.projects import ProjectRecord, resources_for_new

    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    resources = resources_for_new('zeta', base_curation_config())
    record = ProjectRecord(
        slug='zeta',
        display_name='First',
        description='',
        status='building',
        revision=1,
        created_at='t',
        updated_at='t',
        origin=None,
        resources=resources,
    )

    async def _race() -> tuple[Any, ...]:
        return await asyncio.gather(
            lifecycle.write_record(client, record, op_type='create'),
            lifecycle.write_record(client, record, op_type='create'),
            return_exceptions=True,
        )

    results = asyncio.run(_race())
    successes = [r for r in results if not isinstance(r, BaseException)]
    failures = [r for r in results if isinstance(r, BaseException)]
    assert len(successes) == 1, results
    assert len(failures) == 1, results
    assert isinstance(failures[0], HTTPException)
    assert failures[0].detail['error'] == 'slug_taken'


def test_create_project_slug_retired_after_delete() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())
    asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha again'))
    assert exc_info.value.detail['error'] == 'slug_retired'


def test_create_project_capacity_blocked() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)

    async def _blocked(method, url, params=None, **kwargs):
        if url == '/_cluster/health':
            return {'active_shards': 999, 'number_of_data_nodes': 1}
        if url == '/_cluster/settings':
            return {'persistent': {'cluster.max_shards_per_node': 1000}, 'transient': {}}
        if url == '/_nodes/stats/jvm':
            return {'nodes': {'n1': {'jvm': {'mem': {'heap_max_in_bytes': 1024**3}}}}}
        raise NotImplementedError(url)

    client.transport.perform_request = _blocked  # type: ignore[method-assign]
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.create_project(client, slug='overbudget', display_name='x'))
    detail = exc_info.value.detail
    assert detail['error'] == 'shard_budget_exceeded'
    assert detail['capacity'] is not None
    assert 'project:overbudget' not in client.docs  # no project record was written


def test_create_project_capacity_warn_returns_warning() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)

    async def _warn(method, url, params=None, **kwargs):
        if url == '/_cluster/health':
            return {'active_shards': 39, 'number_of_data_nodes': 1}
        if url == '/_cluster/settings':
            return {'persistent': {'cluster.max_shards_per_node': 1000}, 'transient': {}}
        if url == '/_nodes/stats/jvm':
            return {'nodes': {'n1': {'jvm': {'mem': {'heap_max_in_bytes': 2 * 1024**3}}}}}
        raise NotImplementedError(url)

    client.transport.perform_request = _warn  # type: ignore[method-assign]
    record, warnings = asyncio.run(
        lifecycle.create_project(client, slug='nearlimit', display_name='x')
    )
    assert record.status == 'active'
    assert warnings
    assert warnings[0]['code'] == 'shard_budget_high'


def test_create_20_projects_on_roomy_cluster() -> None:
    """No fixed project-count cap (owner D4)."""
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    for i in range(20):
        record, _ = asyncio.run(
            lifecycle.create_project(client, slug=f'proj-{i}', display_name=f'Project {i}')
        )
        assert record.status == 'active'


def test_patch_project_occ() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    record, _ = asyncio.run(lifecycle.create_project(client, slug='cars', display_name='Cars'))
    updated = asyncio.run(
        lifecycle.patch_project(
            client,
            slug='cars',
            display_name='Cars v2',
            description=None,
            expected_revision=record.revision,
        )
    )
    assert updated.display_name == 'Cars v2'
    assert updated.revision == record.revision + 1

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            lifecycle.patch_project(
                client,
                slug='cars',
                display_name='Cars v3',
                description=None,
                expected_revision=record.revision,  # stale
            )
        )
    assert exc_info.value.detail['error'] == 'revision_conflict'


def test_archive_then_write_is_read_only_and_unarchive_restores() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    record, _ = asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(
        lifecycle.create_project(client, slug='beta', display_name='Beta')
    )  # keep >1 active
    archived = asyncio.run(
        lifecycle.archive_project(client, slug='alpha', expected_revision=record.revision)
    )
    assert archived.status == 'archived'

    asyncio.run(registry.ensure_fresh())
    unarchived = asyncio.run(
        lifecycle.unarchive_project(client, slug='alpha', expected_revision=archived.revision)
    )
    assert unarchived.status == 'active'


def test_archive_last_active_project_refused() -> None:
    """``default`` always exists in the snapshot, so archiving the only
    *other* project succeeds (default is still active); archiving
    default next, with nothing else active, is the refusal case."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))
    record, _ = asyncio.run(lifecycle.create_project(client, slug='only', display_name='Only'))
    asyncio.run(lifecycle.archive_project(client, slug='only', expected_revision=record.revision))
    asyncio.run(registry.ensure_fresh())
    default_record = registry.get('default')
    assert default_record is not None
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            lifecycle.archive_project(
                client, slug='default', expected_revision=default_record.revision
            )
        )
    assert exc_info.value.detail['error'] == 'last_active_project'
