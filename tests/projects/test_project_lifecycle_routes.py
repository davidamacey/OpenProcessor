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


@pytest.fixture
def events(monkeypatch):
    """Captures every event published through the real EventHub for this
    test, global and scoped alike."""
    from src.services.curation import event_hub

    monkeypatch.setenv('OP_EVENT_BUS', 'process')
    monkeypatch.setattr(event_hub, '_HUB', None)
    hub = event_hub.get_event_hub()
    captured: list[dict] = []
    real_dispatch = hub._dispatch

    def _recording_dispatch(event):
        captured.append(dict(event))
        real_dispatch(event)

    monkeypatch.setattr(hub, '_dispatch', _recording_dispatch)
    return captured


def test_create_publishes_project_created(events) -> None:
    created = asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='cars', display_name='Cars'))
    )
    assert created.project.status == 'active'
    matches = [e for e in events if e['type'] == 'project.created']
    assert len(matches) == 1
    assert matches[0]['target'] == 'cars'
    assert matches[0]['project'] is None  # global stream, never scoped
    assert matches[0]['status'] == 'active'


def test_patch_archive_unarchive_publish_their_events(events) -> None:
    created = asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='cars', display_name='Cars'))
    )
    asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='keep-active', display_name='x'))
    )
    patched = asyncio.run(
        projects_router.patch_project(
            'cars',
            PatchProjectRequest(display_name='Cars 2', expected_revision=created.project.revision),
        )
    )
    archived = asyncio.run(
        projects_router.archive_project(
            'cars', ArchiveRequest(expected_revision=patched.project.revision)
        )
    )
    asyncio.run(
        projects_router.unarchive_project(
            'cars', ArchiveRequest(expected_revision=archived.project.revision)
        )
    )

    types = [e['type'] for e in events]
    assert types == [
        'project.created',
        'project.created',
        'project.updated',
        'project.archived',
        'project.unarchived',
    ]


def test_delete_finish_publishes_project_deleted(events) -> None:
    """M5 step 9 + m11: drives the real route function (not a stand-in
    for its background task) and awaits the exact task object it
    registers in ``_BACKGROUND_DELETE_TASKS``, proving both that the
    task is kept alive (m11) and that it publishes on completion."""
    from fastapi import Response

    from src.services.projects.registry import get_project_registry

    asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='cars', display_name='Cars'))
    )
    asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='keep-active', display_name='x'))
    )
    asyncio.run(get_project_registry().ensure_fresh())

    async def _delete_flow():
        before = set(projects_router._BACKGROUND_DELETE_TASKS)
        await projects_router.delete_project('cars', Response(), confirm='cars')
        (task,) = projects_router._BACKGROUND_DELETE_TASKS - before
        await task

    asyncio.run(_delete_flow())

    matches = [e for e in events if e['type'] == 'project.deleted']
    assert len(matches) == 1
    assert matches[0]['target'] == 'cars'
    assert matches[0]['status'] == 'deleted'
