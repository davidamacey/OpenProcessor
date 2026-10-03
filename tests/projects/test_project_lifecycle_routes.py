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


def test_create_project_surfaces_keymap_clone_conflicts_as_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Focus-item answer #5: ``create_project`` used to log a create-time
    keymap clone conflict and never surface it -- ``keymap_clone_conflicts``
    on the 201 is always ``[]`` there (that field is the standalone
    ``POST clone_settings`` route's). ``create_project`` already returns
    ``(record, warnings)``, so one ``ProjectWarning`` (code
    ``keymap_clone_conflict``) per dropped conflict must appear in the
    create response's ``warnings`` instead of only the log line."""
    from src.config.project_context import bind_project, try_current_project
    from src.services.curation.keymap import KeymapDoc
    from src.services.projects.registry import get_project_registry

    source = asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='source', display_name='Source'))
    )
    assert source.project.slug == 'source'

    registry = get_project_registry()
    asyncio.run(registry.ensure_fresh())
    source_record = registry.get('source')
    assert source_record is not None
    with bind_project(source_record):
        from src.routers.curation import get_class_registry

        reg = get_class_registry()
        reg.add_class('ice_cream_truck', group='vehicle')
        loaded = reg.load()
        for c in loaded.classes:
            if c.class_name == 'ice_cream_truck':
                c.hotkey_letter = 'i'
        reg._atomic_write(loaded)

    # Fakes the keymap store directly (as the sibling clone_settings test
    # above does) rather than round-tripping a real PUT /keymap, which
    # would itself 409 against the class hotkey just bound above.
    source_doc = KeymapDoc(
        overrides={'cluster.ignore': ['i']}, revision=1, updated_at=None, is_default=False
    )
    default_doc = KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)

    async def _fake_get(_client, _index) -> KeymapDoc:
        current = try_current_project()
        return source_doc if current and current.record.slug == 'source' else default_doc

    async def _fake_save(_client, _index, *, overrides, expected_revision) -> KeymapDoc:
        return KeymapDoc(
            overrides=overrides,
            revision=expected_revision + 1,
            updated_at=None,
            is_default=not overrides,
        )

    monkeypatch.setattr('src.services.curation.keymap.get_keymap_doc', _fake_get)
    monkeypatch.setattr('src.services.curation.keymap.save_keymap_doc', _fake_save)

    # 'classes' clones the source's registry (including the 'i'-bound
    # class) into the new project verbatim, so the 'keymap' axis then
    # validates the source's override against that just-copied class --
    # the naturally-arising create-time conflict, not a synthesized one.
    created = asyncio.run(
        projects_router.create_project(
            CreateProjectRequest(
                slug='newproj',
                display_name='New',
                clone_settings_from='source',
                clone_axes=['classes', 'keymap'],
            )
        )
    )

    assert created.project.status == 'active'
    # create's own field stays empty by design (see focus-item #5) -- the
    # conflict is a ProjectWarning, not this field, on create.
    assert created.keymap_clone_conflicts == []
    warnings = [w.model_dump() for w in created.warnings]
    assert warnings == [
        {
            'code': 'keymap_clone_conflict',
            'message': (
                "keymap action 'cluster.ignore' combo 'i' collides with class "
                "'ice_cream_truck' (id 0) and was dropped from the clone"
            ),
            'detail': {},
        }
    ]


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
        await projects_router.delete_project('cars', Response(), confirm='cars')
        # P3F pass-3 MA1: _BACKGROUND_DELETE_TASKS is now keyed by slug
        # (not a bare set) so a re-DELETE never double-schedules a
        # finish for the same slug -- fetch the exact task the route
        # registered for 'cars'.
        task = projects_router._BACKGROUND_DELETE_TASKS['cars']
        await task

    asyncio.run(_delete_flow())

    matches = [e for e in events if e['type'] == 'project.deleted']
    assert len(matches) == 1
    assert matches[0]['target'] == 'cars'
    assert matches[0]['status'] == 'deleted'
