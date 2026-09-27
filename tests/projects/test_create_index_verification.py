"""P3F item 3 (B2(a) residual): create must never report 'active' with
indexes that don't actually exist.

Two independent guards, per the review's own fix suggestion:
1. ``registry.refresh_strict()`` (raises) replaces the old ``ensure_fresh()``
   (swallows) right after the 'building' write, so a registry that can't
   see the new project yet aborts the create cleanly.
2. A post-``_ensure_indexes`` existence check for every index the record
   claims to own -- ``_ensure_indexes`` is itself fail-open, so #1 alone
   does not close the gap.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch, fake_ensure_indexes, seed_default_project


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


def _registry_for(client: FakeLifecycleOpenSearch) -> ProjectRegistry:
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)
    return registry


def test_transient_registry_refresh_failure_after_building_write_ends_failed(monkeypatch) -> None:
    """The review's own probe: one transient failure into the registry
    refresh that follows the 'building' write must abort the create as
    'failed', never let it proceed to 'active' with zero indexes."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))

    real_refresh_strict = registry.refresh_strict
    calls = {'n': 0}

    async def _flaky_refresh_strict() -> None:
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('simulated transient registry refresh failure')
        await real_refresh_strict()

    monkeypatch.setattr(registry, 'refresh_strict', _flaky_refresh_strict)

    with (
        patch(
            'src.routers.curation._common._ensure_indexes',
            new=AsyncMock(side_effect=fake_ensure_indexes),
        ),
        pytest.raises(RuntimeError, match='simulated transient registry refresh failure'),
    ):
        asyncio.run(lifecycle.create_project(client, slug='gamma', display_name='Gamma'))

    stored = client.docs.get('project:gamma')
    assert stored is not None
    assert stored['status'] == 'failed', 'must abort to failed, never reach active'
    assert not any(name.startswith('op_prj_gamma__') for name in client.indexes), (
        'the aborted create must never have created any of its own indexes'
    )


def test_one_missing_index_after_ensure_indexes_ends_failed_not_active() -> None:
    """_ensure_indexes is itself fail-open (every create/migration
    failure inside it is logged and swallowed) -- simulate it silently
    creating only 6 of 7 indexes, and assert create still ends 'failed',
    never 'active' with a missing index."""
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(seed_default_project(client))

    from src.config.curation import IndexRole, get_curation_config, index_name

    async def _ensure_indexes_missing_one(opensearch) -> None:
        cfg = get_curation_config()
        for role in IndexRole:
            if role == IndexRole.UMAP_VIZ_STATE:
                continue  # simulate this one silently failing to create
            await opensearch.indices.create(index=index_name(cfg, role))

    with (
        patch(
            'src.routers.curation._common._ensure_indexes',
            new=AsyncMock(side_effect=_ensure_indexes_missing_one),
        ),
        pytest.raises(RuntimeError, match='missing indexes'),
    ):
        asyncio.run(lifecycle.create_project(client, slug='gamma', display_name='Gamma'))

    stored = client.docs.get('project:gamma')
    assert stored is not None
    assert stored['status'] == 'failed', 'must abort to failed, never reach active'


def test_real_ensure_indexes_index_create_failure_ends_create_failed() -> None:
    """P3F item 5 (B2(b)): "create fails when index creation fails; fix
    the fakes, not the check." Proves the fake CAN genuinely simulate one
    index's ``indices.create`` raising, through the REAL (unmocked)
    ``_ensure_indexes``/``_create_one`` -- which itself swallows the
    failure and logs it (by design; it never raises) -- and that
    create_project's own post-check (item 3) still catches the missing
    index and ends 'failed', never a partial 'active'."""
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(seed_default_project(client))

    from src.config.curation import IndexRole, get_curation_config, index_name

    real_create = client.indices.create

    async def _create_but_fail_one(*, index: str, body=None):
        cfg = get_curation_config()
        if index == index_name(cfg, IndexRole.UMAP_VIZ_STATE):
            raise RuntimeError('simulated OpenSearch indices.create failure')
        return await real_create(index=index, body=body)

    client.indices.create = _create_but_fail_one  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='missing indexes'):
        asyncio.run(lifecycle.create_project(client, slug='gamma', display_name='Gamma'))

    stored = client.docs.get('project:gamma')
    assert stored is not None
    assert stored['status'] == 'failed', 'must abort to failed, never reach active'
