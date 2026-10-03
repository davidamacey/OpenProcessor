"""P3F item 2 (N1): a 'building' record left by any mid-create failure
must not be wedged forever.

Part A: an exception from ``write_record``'s OWN ``bump_revision`` call
(after the storage-level ``client.index()`` for the 'building' doc
already landed) is caught and the record is flipped to 'failed' --
recoverable through the ordinary DELETABLE_STATUSES path -- instead of
propagating from outside create_project's try block with the record
wedged in 'building' forever.

Part B: the delete-side escape hatch for any OTHER cause of a stuck
'building' record (e.g. a real process crash, which no in-process
exception handler can ever catch): a 'building' record whose
``updated_at`` is stale enough is deletable; a fresh one still 409s.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

from src.services.projects import delete as delete_mod, lifecycle
from src.services.projects.registry import ProjectRegistry, record_to_doc, set_project_registry

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


@pytest.fixture(autouse=True)
def _fake_ensure_indexes_fixture():
    from unittest.mock import AsyncMock, patch

    with patch(
        'src.routers.curation._common._ensure_indexes',
        new=AsyncMock(side_effect=fake_ensure_indexes),
    ):
        yield


def _registry_for(client: FakeLifecycleOpenSearch) -> ProjectRegistry:
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)
    return registry


def test_bump_revision_failure_after_building_write_ends_failed_not_wedged(monkeypatch) -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(seed_default_project(client))  # so 'gamma' isn't the only active project

    from src.services.projects import bootstrap

    real_bump = bootstrap.bump_revision
    calls = {'n': 0}

    async def _flaky_bump(c):
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('simulated registry bump failure')
        return await real_bump(c)

    monkeypatch.setattr(bootstrap, 'bump_revision', _flaky_bump)

    with pytest.raises(RuntimeError, match='simulated registry bump failure'):
        asyncio.run(lifecycle.create_project(client, slug='gamma', display_name='Gamma'))

    stored = client.docs.get('project:gamma')
    assert stored is not None, "the 'building' doc must exist (its own index() succeeded)"
    assert stored['status'] == 'failed', 'N1: must be flipped to failed, never left building'

    # failed is already in _DELETABLE_STATUSES -- a normal DELETE recovers it.
    monkeypatch.setattr(bootstrap, 'bump_revision', real_bump)
    registry = _registry_for(client)
    asyncio.run(registry.ensure_fresh())
    deleting = asyncio.run(lifecycle.delete_project(client, slug='gamma', confirm='gamma'))
    assert deleting.status == 'deleting'
    tombstoned = asyncio.run(lifecycle.delete_project_finish(client, slug='gamma')).record
    assert tombstoned.status == 'deleted'


def _seed_building_record(client: FakeLifecycleOpenSearch, *, slug: str, updated_at: str):
    from src.config.curation import base_curation_config
    from src.config.projects import ProjectRecord, resources_for_new

    resources = resources_for_new(slug, base_curation_config())
    now = datetime.now(UTC).isoformat()
    record = ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='building',
        revision=1,
        created_at=now,
        updated_at=updated_at,
        origin=None,
        resources=resources,
    )
    client.docs[f'project:{slug}'] = record_to_doc(record)
    client._seq[f'project:{slug}'] = client._seq.get(f'project:{slug}', 0) + 1
    client._visible.add(f'project:{slug}')
    return record


def test_stale_building_record_is_deletable() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))
    stale_updated_at = (
        datetime.now(UTC) - timedelta(seconds=delete_mod._BUILDING_STALE_SECONDS + 1)
    ).isoformat()
    record = _seed_building_record(client, slug='stuck', updated_at=stale_updated_at)
    registry._by_slug['stuck'] = record
    asyncio.run(registry.ensure_fresh())

    deleting = asyncio.run(lifecycle.delete_project(client, slug='stuck', confirm='stuck'))
    assert deleting.status == 'deleting'
    # N1's own m2 interaction: pre_delete_status must be 'failed', not
    # 'building' -- a rollback must never recreate the N1 wedge.
    assert deleting.pre_delete_status == 'failed'

    tombstoned = asyncio.run(lifecycle.delete_project_finish(client, slug='stuck')).record
    assert tombstoned.status == 'deleted'


def test_fresh_building_record_delete_still_409s() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))
    fresh_updated_at = datetime.now(UTC).isoformat()
    record = _seed_building_record(client, slug='live', updated_at=fresh_updated_at)
    registry._by_slug['live'] = record
    asyncio.run(registry.ensure_fresh())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project(client, slug='live', confirm='live'))
    assert exc_info.value.detail['error'] == 'invalid_transition'
