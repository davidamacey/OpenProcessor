"""P3: ``GET {prefix}/stats``."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.config.project_context import bind_project
from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry
from src.services.projects.stats import project_stats

from .conftest import FakeLifecycleOpenSearch, fake_ensure_indexes


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
    with patch(
        'src.routers.curation._common._ensure_indexes',
        new=AsyncMock(side_effect=fake_ensure_indexes),
    ):
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


class _TermAwareClient(FakeLifecycleOpenSearch):
    """count() honours a ``term`` query over the seeded item docs."""

    async def count(self, *, index: str, body=None):  # type: ignore[override]
        docs = self.indexes.get(index, [])
        term = ((body or {}).get('query') or {}).get('term')
        if term:
            ((field, value),) = term.items()
            docs = [d for d in docs if d.get(field) == value]
        return {'count': len(docs)}


def test_project_stats_reads_real_holdout_models_and_class_registry(tmp_path, monkeypatch) -> None:
    from src.clients.curation_opensearch.registry import ClassRegistry
    from src.config.curation import items_index

    models = tmp_path / 'models'
    for name, owner in (
        ('cars__det_v1', 'cars'),
        ('dogs__det_v1', 'dogs'),
        ('cars__det_v2', 'cars'),
    ):
        (models / name).mkdir(parents=True)
        (models / name / 'promote.json').write_text(f'{{"project": "{owner}"}}')
    monkeypatch.setattr(
        'src.services.training.triton_repo.resolve_triton_models_dir', lambda: models
    )

    client = _TermAwareClient()
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)
    asyncio.run(lifecycle.create_project(client, slug='cars', display_name='Cars'))
    asyncio.run(registry.ensure_fresh())
    record = registry.get('cars')
    assert record is not None
    reg = ClassRegistry(path=record.resources.class_registry_path)
    for name in ('car', 'truck', 'van'):
        reg.add_class(name)

    with bind_project(record):
        client.indexes[items_index()] = [
            {'item_id': '1', 'test_holdout': True},
            {'item_id': '2', 'test_holdout': True},
            {'item_id': '3'},
        ]
        counts = asyncio.run(project_stats(client))['counts']

    assert counts['holdout_items'] == 2
    assert counts['promoted_models'] == 2
    assert counts['classes'] == 3
