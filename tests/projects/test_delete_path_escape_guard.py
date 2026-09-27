"""P3F item 6 m1: the delete path guard for ``train_jobs_dir``/
``autolabel_dir`` used to be checked against a root DERIVED FROM the
same path (``path.parent.parent``), which can never refuse anything --
any path is trivially "within" its own grandparent. A corrupted or
hand-edited registry doc pointing either field outside the real shared
root must now be refused with ``path_escape``, not silently followed.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from fastapi import HTTPException

from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, record_to_doc, set_project_registry

from .conftest import FakeLifecycleOpenSearch, seed_default_project


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'jobs'))
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


def test_delete_refuses_path_escape_for_corrupted_train_jobs_dir(tmp_path) -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(registry.ensure_fresh())

    record = registry.get('alpha')
    assert record is not None
    escaped_dir = tmp_path / 'escaped_outside_root'
    escaped_dir.mkdir(parents=True)
    (escaped_dir / 'canary.txt').write_text('do not delete me', encoding='utf-8')
    corrupted = replace(record, resources=replace(record.resources, train_jobs_dir=escaped_dir))
    client.docs['project:alpha'] = record_to_doc(corrupted)
    client._seq['project:alpha'] = client._seq.get('project:alpha', 0) + 1
    registry._by_slug['alpha'] = corrupted

    deleting = asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert exc_info.value.detail['error'] == 'path_escape'
    assert exc_info.value.status_code == 500
    assert escaped_dir.is_dir(), 'the escaped directory must survive, untouched'
    assert (escaped_dir / 'canary.txt').is_file()
