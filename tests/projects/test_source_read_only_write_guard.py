"""M1b regression (W3/W4 review 2026-09-28, round 2): a planted write
attempt inside each ``bind_project(source, read_only=True)`` scope must be
rejected by the guard.

The round-1 M1b test (``test_prompt_packs_router.py::
test_from_project_clone_is_read_only_and_never_writes_to_source`` and this
module's sibling ``test_clone_settings.py::
test_clone_source_stays_byte_identical``) only prove that the CURRENT
(correct) clone code never issues a write while the source is bound --
they never plant a write, so removing ``read_only=True`` from any of the
five source binds (``clone_shared.py``'s ``read_source_record`` plus the
four in ``clone.py``: ``settings_defaults``, ``keymap``,
``_clone_prompt_packs``, ``_clone_activations``) leaves the whole suite
green (confirmed by the round-2 reviewer). These tests close that gap by
making a read step *inside* each bind also attempt a real write via the
guard's own ``check_request`` -- mirroring exactly what
``ProjectGuardedTransport`` does for a live OpenSearch client -- and
asserting ``ProjectReadOnly`` is what stops it.

Manually confirmed RED (this file only) when ``read_only=True`` is
removed from each of the five binds in turn; reverted after confirming.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.config.project_context import try_current_project
from src.services.projects import lifecycle
from src.services.projects.guard import ProjectReadOnly, check_request
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


def _registry_for(client: FakeLifecycleOpenSearch) -> ProjectRegistry:
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)
    return registry


def _plant_write_probe() -> None:
    """Attempt a write against whatever project is currently bound, via
    the real guard's ``check_request`` -- the same function a live
    OpenSearch client's ``ProjectGuardedTransport`` calls on every
    request. Raises ``ProjectReadOnly`` when the guard is doing its job;
    silently returns (proving the write would have gone through) when it
    isn't."""
    bound = try_current_project()
    assert bound is not None, 'probe called with no project bound'
    index = next(iter(bound.record.resources.indexes.values()))
    check_request(
        'PUT',
        f'/{index}/_doc/m1b_probe',
        {'planted': True},
        {bound.record.slug: bound.record},
    )


async def _seed(client: FakeLifecycleOpenSearch) -> tuple[Any, Any]:
    registry = _registry_for(client)
    await lifecycle.create_project(client, slug='source', display_name='Source')
    await lifecycle.create_project(client, slug='target', display_name='Target')
    await registry.ensure_fresh()
    source = registry.get('source')
    target = registry.get('target')
    assert source is not None
    assert target is not None
    return source, target


def test_settings_defaults_source_bind_blocks_planted_write(monkeypatch) -> None:
    """``clone.py:167`` -- ``bind_project(source, read_only=True)`` around
    ``get_curation_settings``."""
    client = FakeLifecycleOpenSearch()
    _source, target = asyncio.run(_seed(client))

    async def _probing_get_settings(_client: Any) -> dict[str, Any]:
        _plant_write_probe()
        return {'defaults': {}}

    monkeypatch.setattr(
        'src.clients.curation_opensearch.get_curation_settings', _probing_get_settings
    )
    with pytest.raises(ProjectReadOnly):
        asyncio.run(
            lifecycle.clone_settings(
                client, target_record=target, from_slug='source', axes=['settings_defaults']
            )
        )


def test_keymap_source_bind_blocks_planted_write(monkeypatch) -> None:
    """``clone.py:198`` -- ``bind_project(source, read_only=True)`` around
    ``get_keymap_doc``."""
    client = FakeLifecycleOpenSearch()
    _source, target = asyncio.run(_seed(client))

    async def _probing_get_keymap_doc(_client: Any, _index: str) -> Any:
        _plant_write_probe()
        from src.services.curation.keymap import KeymapDoc

        return KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)

    monkeypatch.setattr('src.services.curation.keymap.get_keymap_doc', _probing_get_keymap_doc)
    with pytest.raises(ProjectReadOnly):
        asyncio.run(
            lifecycle.clone_settings(
                client, target_record=target, from_slug='source', axes=['keymap']
            )
        )


def test_clone_prompt_packs_source_bind_blocks_planted_write(monkeypatch) -> None:
    """``clone.py:293`` (``_clone_prompt_packs``) -- ``bind_project(source,
    read_only=True)`` around the source config store's ``ensure_fresh``."""
    client = FakeLifecycleOpenSearch()
    _source, target = asyncio.run(_seed(client))

    from src.services.config_store.store import ConfigStore

    orig_ensure_fresh = ConfigStore.ensure_fresh

    async def _probing_ensure_fresh(self: ConfigStore, probe_client: Any, *a: Any, **kw: Any):
        bound = try_current_project()
        if bound is not None and bound.record.slug == 'source':
            _plant_write_probe()
        return await orig_ensure_fresh(self, probe_client, *a, **kw)

    monkeypatch.setattr(ConfigStore, 'ensure_fresh', _probing_ensure_fresh)
    with pytest.raises(ProjectReadOnly):
        asyncio.run(
            lifecycle.clone_settings(
                client, target_record=target, from_slug='source', axes=['prompt_packs']
            )
        )


def test_clone_activations_source_bind_blocks_planted_write(monkeypatch) -> None:
    """``clone.py:360`` (``_clone_activations``) -- ``bind_project(source,
    read_only=True)`` around ``get_activation``."""
    client = FakeLifecycleOpenSearch()
    _source, target = asyncio.run(_seed(client))

    from src.services.config_store.index import get_activation as orig_get_activation

    async def _probing_get_activation(probe_client: Any, index: str, axis: Any):
        bound = try_current_project()
        if bound is not None and bound.record.slug == 'source':
            _plant_write_probe()
        return await orig_get_activation(probe_client, index, axis)

    monkeypatch.setattr('src.services.config_store.index.get_activation', _probing_get_activation)
    with pytest.raises(ProjectReadOnly):
        asyncio.run(
            lifecycle.clone_settings(
                client, target_record=target, from_slug='source', axes=['activations']
            )
        )


def test_read_source_record_bind_blocks_planted_write(monkeypatch) -> None:
    """``clone_shared.py``'s ``read_source_record`` -- the shared
    cross-project resolver both prompt-pack and region-profile clone
    routers use."""
    client = FakeLifecycleOpenSearch()
    _source, _target = asyncio.run(_seed(client))

    from src.services.config_store.clone_shared import read_source_record

    async def _resolve_and_plant(_opensearch: Any) -> str:
        _plant_write_probe()
        return 'never reached'

    with pytest.raises(ProjectReadOnly):
        asyncio.run(
            read_source_record(
                from_project='source',
                target_slug='target',
                opensearch=client,
                resolve=_resolve_and_plant,
            )
        )
