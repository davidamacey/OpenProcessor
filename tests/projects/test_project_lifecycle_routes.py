"""P3 delta 2: every lifecycle mutation answers
``{"project": ProjectSummary, "warnings": [...]}``, and ``ProjectSummary``
carries ``revision`` so a client can PATCH/clone_settings with
``expected_revision`` straight from the create response."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.core.exceptions import RequestRejectedError
from src.routers.curation import projects as projects_router
from src.schemas.curation.projects import (
    ArchiveRequest,
    CloneSettingsRequest,
    CreateProjectRequest,
    PatchProjectRequest,
)
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch, fake_ensure_indexes


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation_accessors as curation_mod
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
        'src.services.curation.index_bootstrap.ensure_indexes',
        new=AsyncMock(side_effect=fake_ensure_indexes),
    ):
        yield


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    client = FakeLifecycleOpenSearch()
    registry = ProjectRegistry()
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

    with pytest.raises(RequestRejectedError) as exc_info:
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
    """A dropped keymap-clone conflict is served on the route
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

    async def _fake_get() -> KeymapDoc:
        current = try_current_project()
        return source_doc if current and current.record.slug == 'source' else target_doc

    async def _fake_save(*, overrides, expected_revision) -> KeymapDoc:
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

        get_class_registry().seed(['ice_cream_truck'], group='vehicle', hotkey_letter='i')

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

        get_class_registry().seed(['ice_cream_truck'], group='vehicle', hotkey_letter='i')

    # Fakes the keymap store directly (as the sibling clone_settings test
    # above does) rather than round-tripping a real PUT /keymap, which
    # would itself 409 against the class hotkey just bound above.
    source_doc = KeymapDoc(
        overrides={'cluster.ignore': ['i']}, revision=1, updated_at=None, is_default=False
    )
    default_doc = KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)

    async def _fake_get() -> KeymapDoc:
        current = try_current_project()
        return source_doc if current and current.record.slug == 'source' else default_doc

    async def _fake_save(*, overrides, expected_revision) -> KeymapDoc:
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
                "'ice_cream_truck' (id 0) so the keymap was not cloned"
            ),
            'detail': {},
        }
    ]


@pytest.fixture
def events(captured_events):
    """Every event published during the test, global and scoped alike."""
    return captured_events


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


class _StubContext:
    """The four members of a ``JobContext`` the delete handler touches."""

    def __init__(self, slug: str, client) -> None:
        self.params = {'slug': slug}
        self._client = client

    async def client(self, _name: str):
        return self._client

    async def checkpoint(self, _cursor) -> None:
        return None

    async def progress(self, *_args, **_kwargs) -> None:
        return None


def test_delete_finish_publishes_project_deleted(events, producer) -> None:
    """The DELETE route queues the ``project_delete`` job; the job's handler runs the finish and
    publishes ``project.deleted`` once the tombstone is written."""
    from fastapi import Response

    from src.db.session import async_session_factory
    from src.services.projects.registry import get_project_registry
    from src.tasks import jobs_c

    asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='cars', display_name='Cars'))
    )
    asyncio.run(
        projects_router.create_project(CreateProjectRequest(slug='keep-active', display_name='x'))
    )
    asyncio.run(get_project_registry().ensure_fresh())

    async def _delete_flow():
        async with async_session_factory()() as session:
            accepted = await projects_router.delete_project(
                'cars', Response(), session, confirm='cars'
            )
        assert accepted.job_id == producer.sent()[0]
        assert not [e for e in events if e['type'] == 'project.deleted']
        client = await projects_router.make_curation_opensearch()
        await jobs_c.project_delete(_StubContext('cars', client))

    asyncio.run(_delete_flow())

    matches = [e for e in events if e['type'] == 'project.deleted']
    assert len(matches) == 1
    assert matches[0]['target'] == 'cars'
    assert matches[0]['status'] == 'deleted'
