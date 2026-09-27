"""P3F item 7 / P1R minor 10: if the most recent registry refresh
failed (OpenSearch flaky), the request-binder must not trust a
possibly-stale ``active`` status -- bind read-only rather than let a
write through against a project that may have flipped to ``deleting``
on another instance in the meantime.
"""

from __future__ import annotations

import asyncio

import pytest

from src.config.project_context import try_current_project
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeRegistryOpenSearch, seed_default_project


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation as curation_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    set_project_registry(None)
    yield
    curation_mod._default_curation_config = None
    set_project_registry(None)


def test_binds_read_only_when_last_refresh_failed() -> None:
    from src.routers.curation._project_deps import _resolve_and_bind

    async def _run() -> None:
        client = FakeRegistryOpenSearch()
        await seed_default_project(client)

        registry = ProjectRegistry(lambda: client)
        await registry.ensure_fresh()  # succeeds once; default is now known active
        assert not registry.stale

        # Simulate a flaky OpenSearch on the *next* refresh: ensure_fresh
        # swallows the error and keeps the last snapshot, but the
        # registry must now report itself stale.
        async def _broken_client():
            raise ConnectionError('opensearch unreachable')

        registry._client_factory = _broken_client
        await registry.ensure_fresh()
        assert registry.stale

        set_project_registry(registry)
        record = await _resolve_and_bind('default')
        assert record.status == 'active'  # the stale snapshot still says active...
        bound = try_current_project()
        assert bound is not None
        assert bound.read_only is True  # ...but the bind must not trust it

    asyncio.run(_run())


def test_binds_writable_when_registry_is_fresh() -> None:
    from src.routers.curation._project_deps import _resolve_and_bind

    async def _run() -> None:
        client = FakeRegistryOpenSearch()
        await seed_default_project(client)
        registry = ProjectRegistry(lambda: client)
        await registry.ensure_fresh()
        assert not registry.stale
        set_project_registry(registry)

        await _resolve_and_bind('default')
        bound = try_current_project()
        assert bound is not None
        assert bound.read_only is False

    asyncio.run(_run())
