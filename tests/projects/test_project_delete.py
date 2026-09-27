"""P3: guarded delete -- dry-run, real (background-completing, delta 10),
`default` protection (D5), confirm mismatch, idempotent re-run,
slug retirement."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch, seed_default_project


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
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


@pytest.fixture(autouse=True)
def _noop_ensure_indexes():
    with patch('src.routers.curation._common._ensure_indexes', new=AsyncMock()):
        yield


def _registry_for(client: FakeLifecycleOpenSearch) -> ProjectRegistry:
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)
    return registry


def test_dry_run_lists_report_and_writes_nothing() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    docs_before = dict(client.docs)

    report = asyncio.run(lifecycle.dry_run_delete(client, slug='alpha'))
    assert report['indexes']
    assert report['dirs']
    assert report['blocking'] == []
    assert client.docs == docs_before  # dry run writes nothing


def test_delete_default_is_protected() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(seed_default_project(client))
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project(client, slug='default', confirm='default'))
    assert exc_info.value.detail['error'] == 'project_protected'

    dry_run = asyncio.run(lifecycle.dry_run_delete(client, slug='default'))
    assert 'project_protected' in dry_run['blocking']


def test_delete_confirm_mismatch() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='not-alpha'))
    assert exc_info.value.detail['error'] == 'confirm_mismatch'


def test_delete_last_active_project_refused() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    default_record = asyncio.run(seed_default_project(client))
    asyncio.run(lifecycle.create_project(client, slug='only', display_name='Only'))
    asyncio.run(
        lifecycle.archive_project(client, slug='default', expected_revision=default_record.revision)
    )
    asyncio.run(registry.ensure_fresh())
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project(client, slug='only', confirm='only'))
    assert exc_info.value.detail['error'] == 'last_active_project'


def test_delete_removes_exact_indexes_and_tombstones_slug() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())
    alpha_record = registry.get('alpha')
    assert alpha_record is not None
    expected_indexes = set(alpha_record.resources.indexes.values())

    deleting = asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    tombstoned = asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert tombstoned.status == 'deleted'
    assert set(client.deleted_indexes) == expected_indexes  # exact names, never a pattern

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha again'))
    assert exc_info.value.detail['error'] == 'slug_retired'


def test_delete_finish_is_idempotent_after_crash_between_steps() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())

    asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    first = asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert first.status == 'deleted'
    # A crash could leave delete_project_finish re-run against an
    # already-deleted record; it must be a no-op, not an error.
    second = asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert second.status == 'deleted'
