"""P3F item 4 (M5 step 4): delete unloads the project's own promoted
models, and the dry run actually reports them (previously hardcoded
``[]`` even when the project owned promoted, shared models).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry
from src.services.training import triton_promote

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


def test_dry_run_reports_owned_promoted_model(tmp_path) -> None:
    _write_promote_json(tmp_path, 'cars__detector_v1', project='cars', shared=True)

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        await lifecycle.create_project(client, slug='cars', display_name='Cars')

        report = await lifecycle.dry_run_delete(client, slug='cars')
        assert report['promoted_models'] == ['cars__detector_v1']

    asyncio.run(_run())


def test_dry_run_reports_a_private_promoted_model_too(tmp_path) -> None:
    """P3F pass-3 MA2: promoted_models must report EVERY model this
    project owns, private (shared=False, the common case) included --
    previously this enumerated only `_shared_model_users` (the
    `is_model_shared` subset), so a private-only project always
    reported ``[]`` here even though it owned a promoted model."""
    _write_promote_json(tmp_path, 'cars__private_v1', project='cars', shared=False)

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        await lifecycle.create_project(client, slug='cars', display_name='Cars')

        report = await lifecycle.dry_run_delete(client, slug='cars')
        assert report['promoted_models'] == ['cars__private_v1']

    asyncio.run(_run())


def test_real_delete_unloads_owned_promoted_model(tmp_path, monkeypatch) -> None:
    """force=True bypasses the in_use block (already covered by
    test_delete_shared_model_in_use.py); this proves the finish step
    actually calls the unload primitive for the owned model, never a
    real Triton call."""
    _write_promote_json(tmp_path, 'cars__detector_v1', project='cars', shared=True)

    unload_calls: list[str] = []

    async def _fake_unload(triton_name: str, *, promoter=None):
        unload_calls.append(triton_name)
        return triton_promote.UnloadResult(
            triton_name=triton_name, triton_unloaded=True, directory_removed=True
        )

    # _unload_owned_models does `from ...triton_promote import
    # unload_triton_model` inside its own function body (a local import,
    # re-resolved on every call), so the fake must replace the name in
    # its DEFINING module, not delete.py's (which never binds it at
    # module scope).
    monkeypatch.setattr(triton_promote, 'unload_triton_model', AsyncMock(side_effect=_fake_unload))

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        await lifecycle.create_project(client, slug='cars', display_name='Cars')
        await lifecycle.create_project(client, slug='dogs', display_name='Dogs')

        deleting = await lifecycle.delete_project(client, slug='cars', confirm='cars', force=True)
        assert deleting.status == 'deleting'
        finished = await lifecycle.delete_project_finish(client, slug='cars')
        assert finished.status == 'deleted'

    asyncio.run(_run())
    assert unload_calls == ['cars__detector_v1']


def test_real_delete_unloads_a_private_promoted_model_with_no_force(tmp_path, monkeypatch) -> None:
    """P3F pass-3 MA2: a project with ONLY private (non-shared)
    promoted models must have them unloaded on a NORMAL delete (no
    ``force`` needed -- private models never trip the ``in_use``
    refusal in the first place, so the old
    ``_shared_model_users``-only wiring meant unload never ran for
    this, the common, case)."""
    _write_promote_json(tmp_path, 'cars__private_v1', project='cars', shared=False)

    unload_calls: list[str] = []

    async def _fake_unload(triton_name: str, *, promoter=None):
        unload_calls.append(triton_name)
        return triton_promote.UnloadResult(
            triton_name=triton_name, triton_unloaded=True, directory_removed=True
        )

    monkeypatch.setattr(triton_promote, 'unload_triton_model', AsyncMock(side_effect=_fake_unload))

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        await lifecycle.create_project(client, slug='cars', display_name='Cars')
        await lifecycle.create_project(client, slug='dogs', display_name='Dogs')

        # No force=True: a private model must never trip `in_use`.
        deleting = await lifecycle.delete_project(client, slug='cars', confirm='cars')
        assert deleting.status == 'deleting'
        finished = await lifecycle.delete_project_finish(client, slug='cars')
        assert finished.status == 'deleted'

    asyncio.run(_run())
    assert unload_calls == ['cars__private_v1']
