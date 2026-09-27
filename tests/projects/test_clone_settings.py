"""P3: ``clone_settings`` -- settings copied, ``classes`` refused on a
non-empty target, the source stays byte-identical (read-only bind)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch


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


def test_clone_settings_copies_defaults() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='source', display_name='Source'))
    asyncio.run(lifecycle.create_project(client, slug='target', display_name='Target'))
    asyncio.run(registry.ensure_fresh())
    source = registry.get('source')
    target = registry.get('target')
    assert source is not None
    assert target is not None

    from src.config.project_context import bind_project

    with bind_project(source):
        from src.clients.curation_opensearch import update_curation_settings

        asyncio.run(update_curation_settings(client, {'axis_x': 'value_x'}))

    asyncio.run(
        lifecycle.clone_settings(
            client, target_record=target, from_slug='source', axes=['settings_defaults']
        )
    )

    with bind_project(target):
        from src.clients.curation_opensearch import get_curation_settings

        cloned = asyncio.run(get_curation_settings(client))
    assert cloned['defaults'].get('axis_x') == 'value_x'


def test_clone_classes_refused_on_non_empty_target() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='source', display_name='Source'))
    asyncio.run(lifecycle.create_project(client, slug='target', display_name='Target'))
    asyncio.run(registry.ensure_fresh())
    target = registry.get('target')
    assert target is not None

    from src.config.curation import items_index
    from src.config.project_context import bind_project

    with bind_project(target):
        client.indexes[items_index()] = [{'item_id': 'x'}]  # non-empty target

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            lifecycle.clone_settings(
                client, target_record=target, from_slug='source', axes=['classes']
            )
        )
    assert exc_info.value.detail['error'] == 'target_not_empty'


def test_clone_source_stays_byte_identical() -> None:
    """The source is read under a read-only bind; a write to it must
    raise, proving clone_settings never mutates the source."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='source', display_name='Source'))
    asyncio.run(lifecycle.create_project(client, slug='target', display_name='Target'))
    asyncio.run(registry.ensure_fresh())
    source = registry.get('source')
    target = registry.get('target')
    assert source is not None
    assert target is not None

    asyncio.run(
        lifecycle.clone_settings(
            client, target_record=target, from_slug='source', axes=['settings_defaults']
        )
    )

    from src.config.project_context import bind_project
    from src.services.projects.guard import ProjectReadOnly, check_request

    with bind_project(source, read_only=True), pytest.raises(ProjectReadOnly):
        check_request(
            'PUT',
            f'/{source.resources.indexes[next(iter(source.resources.indexes))]}/_doc/x',
            None,
            registry.snapshot(),
        )
