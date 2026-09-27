"""P3 delta 2: every lifecycle mutation answers
``{"project": ProjectSummary, "warnings": [...]}``, and ``ProjectSummary``
carries ``revision`` so a client can PATCH/clone_settings with
``expected_revision`` straight from the create response."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.routers.curation import projects as projects_router
from src.routers.curation._project_models import (
    ArchiveRequest,
    CloneSettingsRequest,
    CreateProjectRequest,
    PatchProjectRequest,
)
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


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    client = FakeLifecycleOpenSearch()
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)

    async def _fake_make_client():
        return client

    monkeypatch.setattr(projects_router, 'make_curation_opensearch', _fake_make_client)
    return client


def test_create_route_envelope() -> None:
    response = asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='cars', display_name='Cars'))
    )
    assert response.project.slug == 'cars'
    assert response.project.status == 'active'
    assert response.project.revision >= 1
    assert isinstance(response.warnings, list)


def test_patch_route_envelope_and_revision_conflict() -> None:
    created = asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='cars', display_name='Cars'))
    )
    patched = asyncio.run(
        projects_router.patch_project(
            'cars',
            PatchProjectRequest(display_name='Cars 2', expected_revision=created.project.revision),
        )
    )
    assert patched.project.display_name == 'Cars 2'
    assert patched.project.revision == created.project.revision + 1

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            projects_router.patch_project(
                'cars',
                PatchProjectRequest(
                    display_name='Cars 3', expected_revision=created.project.revision
                ),
            )
        )
    assert exc_info.value.detail['error'] == 'revision_conflict'


def test_archive_unarchive_route_envelope() -> None:
    created = asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='cars', display_name='Cars'))
    )
    asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='keep-active', display_name='x'))
    )
    archived = asyncio.run(
        projects_router.archive_project(
            'cars', ArchiveRequest(expected_revision=created.project.revision)
        )
    )
    assert archived.project.status == 'archived'
    unarchived = asyncio.run(
        projects_router.unarchive_project(
            'cars', ArchiveRequest(expected_revision=archived.project.revision)
        )
    )
    assert unarchived.project.status == 'active'


def test_clone_settings_route_serves_keymap_conflicts_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W2b: a dropped keymap-clone conflict is served on the route
    response as a structured ``keymap_clone_conflicts`` list (action_id,
    combo, class), never silently swallowed."""
    from src.config.project_context import try_current_project
    from src.services.curation.keymap import KeymapDoc

    source = asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='source', display_name='Source'))
    )
    target = asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='target', display_name='Target'))
    )

    source_doc = KeymapDoc(
        overrides={'cluster.ignore': ['i']}, revision=1, updated_at=None, is_default=False
    )
    target_doc = KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)

    async def _fake_get(_client, _index) -> KeymapDoc:
        current = try_current_project()
        return source_doc if current and current.record.slug == 'source' else target_doc

    async def _fake_save(_client, _index, *, overrides, expected_revision) -> KeymapDoc:
        return KeymapDoc(
            overrides=overrides,
            revision=expected_revision + 1,
            updated_at=None,
            is_default=not overrides,
        )

    monkeypatch.setattr('src.services.curation.keymap.get_keymap_doc', _fake_get)
    monkeypatch.setattr('src.services.curation.keymap.save_keymap_doc', _fake_save)

    from src.config.project_context import bind_project
    from src.services.projects.registry import get_project_registry

    registry = get_project_registry()
    asyncio.run(registry.ensure_fresh())
    target_record = registry.get('target')
    assert target_record is not None
    with bind_project(target_record):
        from src.routers.curation import get_class_registry

        reg = get_class_registry()
        reg.add_class('ice_cream_truck', group='vehicle')
        loaded = reg.load()
        for c in loaded.classes:
            if c.class_name == 'ice_cream_truck':
                c.hotkey_letter = 'i'
        reg._atomic_write(loaded)

    response = asyncio.run(
        projects_router.clone_settings_route(
            'target',
            CloneSettingsRequest(
                **{'from': 'source'}, axes=['keymap'], expected_revision=target.project.revision
            ),
        )
    )
    assert [c.model_dump() for c in response.keymap_clone_conflicts] == [
        {
            'action_id': 'cluster.ignore',
            'combo': 'i',
            'class_id': 0,
            'class_name': 'ice_cream_truck',
        }
    ]
    assert source.project.slug == 'source'
