"""P3: ``GET {prefix}/stats``."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.config.project_context import bind_project
from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry
from src.services.projects.stats import project_stats

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


def test_project_stats_counts_and_disk() -> None:
    client = FakeLifecycleOpenSearch()
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)
    asyncio.run(lifecycle.create_project(client, slug='cars', display_name='Cars'))
    asyncio.run(registry.ensure_fresh())
    record = registry.get('cars')
    assert record is not None

    from src.config.curation import items_index

    with bind_project(record):
        client.indexes[items_index()] = [{'item_id': '1'}, {'item_id': '2'}]
        stats = asyncio.run(project_stats(client))

    assert stats['counts']['items'] == 2
    assert 'disk' in stats
    assert stats['jobs']['running'] == []
    assert any(entry['name'] for entry in stats['indexes'])
