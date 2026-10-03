"""P3F item 2: deleting a project that owns a cross-project-shared
promoted model (projects_plan.md §5.5) is refused (409 ``in_use``)
unless ``force=True``, in which case the bypass is logged distinctly.

Sharing on for one of the project's own models blocks the delete; the 409
names the models and the projects whose ACTIVE detection profile uses them;
``force`` still proceeds.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException

from src.services.projects import delete as delete_mod, lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch, seed_default_project


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path / 'triton_models'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)
    yield
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


def _write_promote_json(tmp_path, model_name: str, *, project: str, shared: bool) -> None:
    model_dir = tmp_path / 'triton_models' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'promote.json').write_text(
        json.dumps({'project': project, 'shared': shared}), encoding='utf-8'
    )


def test_delete_refuses_when_project_owns_a_shared_model(tmp_path) -> None:
    _write_promote_json(tmp_path, 'cars__detector_v1', project='cars', shared=True)

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        # P3F item 3 (B2(a) residual): create_project's registry
        # refresh_strict() needs a real, reachable registry bound during
        # the create itself -- set it before create_project runs.
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        await lifecycle.create_project(client, slug='cars', display_name='Cars')
        await lifecycle.create_project(client, slug='dogs', display_name='Dogs')

        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.delete_project(client, slug='cars', confirm='cars')
        detail = exc_info.value.detail
        assert detail['error'] == 'in_use'
        assert 'cars__detector_v1' in detail['message']
        assert detail['projects'] == []

    asyncio.run(_run())


def test_delete_force_bypasses_shared_model_block_and_logs(tmp_path, monkeypatch) -> None:
    _write_promote_json(tmp_path, 'cars__detector_v1', project='cars', shared=True)

    warnings: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        delete_mod.logger,
        'warning',
        lambda event, **fields: warnings.append((event, fields)),
    )

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        # P3F item 3 (B2(a) residual): create_project's registry
        # refresh_strict() needs a real, reachable registry bound during
        # the create itself -- set it before create_project runs.
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        await lifecycle.create_project(client, slug='cars', display_name='Cars')
        await lifecycle.create_project(client, slug='dogs', display_name='Dogs')

        record = await lifecycle.delete_project(client, slug='cars', confirm='cars', force=True)
        assert record.status == 'deleting'

    asyncio.run(_run())

    events = [event for event, _ in warnings]
    assert 'project_delete_forced_past_shared_models' in events
    fields = next(f for e, f in warnings if e == 'project_delete_forced_past_shared_models')
    assert fields['project'] == 'cars'
    assert fields['shared_models'] == ['cars__detector_v1']


def test_delete_allowed_when_project_has_no_shared_models(tmp_path) -> None:
    _write_promote_json(tmp_path, 'cars__detector_v1', project='cars', shared=False)

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        # P3F item 3 (B2(a) residual): create_project's registry
        # refresh_strict() needs a real, reachable registry bound during
        # the create itself -- set it before create_project runs.
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        await lifecycle.create_project(client, slug='cars', display_name='Cars')
        await lifecycle.create_project(client, slug='dogs', display_name='Dogs')

        record = await lifecycle.delete_project(client, slug='cars', confirm='cars')
        assert record.status == 'deleting'

    asyncio.run(_run())


def test_delete_refusal_names_the_project_whose_active_profile_uses_the_model(
    tmp_path, monkeypatch
) -> None:
    _write_promote_json(tmp_path, 'cars__detector_v1', project='cars', shared=True)

    async def _users(_client, model_name):
        assert model_name == 'cars__detector_v1'
        return [('dogs', 'dog_profile'), ('cars', 'own_profile')]

    monkeypatch.setattr('src.services.config_store.project_usage.active_detector_users', _users)

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        await lifecycle.create_project(client, slug='cars', display_name='Cars')
        await lifecycle.create_project(client, slug='dogs', display_name='Dogs')

        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.delete_project(client, slug='cars', confirm='cars')
        detail = exc_info.value.detail
        assert detail['error'] == 'in_use'
        assert detail['projects'] == ['dogs']
        assert detail['used_by'] == [{'project': 'dogs', 'profile': 'dog_profile'}]

    asyncio.run(_run())


def test_delete_fails_closed_when_another_project_cannot_be_read(tmp_path, monkeypatch) -> None:
    _write_promote_json(tmp_path, 'cars__detector_v1', project='cars', shared=True)

    async def _broken(_client, _model_name):
        raise ConnectionError('opensearch down')

    monkeypatch.setattr('src.services.config_store.project_usage.active_detector_users', _broken)

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        await lifecycle.create_project(client, slug='cars', display_name='Cars')
        await lifecycle.create_project(client, slug='dogs', display_name='Dogs')

        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.delete_project(client, slug='cars', confirm='cars')
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail['error'] == 'config_store_unavailable'

    asyncio.run(_run())
