"""P3: guarded delete -- dry-run, real (background-completing, delta 10),
`default` protection (D5), confirm mismatch, idempotent re-run,
slug retirement."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

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


def test_dry_run_lists_report_and_writes_nothing() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    docs_before = dict(client.docs)

    report = asyncio.run(lifecycle.dry_run_delete(client, slug='alpha'))
    assert report['indexes']
    assert report['dirs']
    assert report['blocking'] == []
    assert client.docs == docs_before  # dry run writes nothing


def test_delete_default_is_protected() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(seed_default_project(client))
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project(client, slug='default', confirm='default'))
    assert exc_info.value.detail['error'] == 'project_protected'

    dry_run = asyncio.run(lifecycle.dry_run_delete(client, slug='default'))
    assert 'project_protected' in dry_run['blocking']


def test_delete_confirm_mismatch() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='not-alpha'))
    assert exc_info.value.detail['error'] == 'confirm_mismatch'


def test_delete_last_active_project_refused() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    default_record = asyncio.run(seed_default_project(client))
    asyncio.run(lifecycle.create_project(client, slug='only', display_name='Only'))
    asyncio.run(
        lifecycle.archive_project(client, slug='default', expected_revision=default_record.revision)
    )
    asyncio.run(registry.ensure_fresh())
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project(client, slug='only', confirm='only'))
    assert exc_info.value.detail['error'] == 'last_active_project'


def test_delete_removes_exact_indexes_and_tombstones_slug() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())
    alpha_record = registry.get('alpha')
    assert alpha_record is not None
    expected_indexes = set(alpha_record.resources.indexes.values())

    deleting = asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    tombstoned = asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert tombstoned.status == 'deleted'
    assert set(client.deleted_indexes) == expected_indexes  # exact names, never a pattern

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha again'))
    assert exc_info.value.detail['error'] == 'slug_retired'


def test_delete_finish_is_idempotent_after_crash_between_steps() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())

    asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    first = asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert first.status == 'deleted'
    # A crash could leave delete_project_finish re-run against an
    # already-deleted record; it must be a no-op, not an error.
    second = asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert second.status == 'deleted'


def test_delete_refuses_from_building_or_deleting(monkeypatch) -> None:
    """m3: the plan's machine is active|archived|failed -> deleting. A
    delete racing a create (still 'building') or a second delete
    (already 'deleting') must be refused, not resurrect/re-flip it."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())

    from dataclasses import replace

    from src.services.projects.registry import record_to_doc

    alpha_record = registry.get('alpha')
    assert alpha_record is not None
    building = replace(alpha_record, status='building')
    client.docs['project:alpha'] = record_to_doc(building)
    client._seq['project:alpha'] = client._seq.get('project:alpha', 0) + 1
    registry._by_slug['alpha'] = building  # _resolve_existing reads the in-memory snapshot

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    assert exc_info.value.detail['error'] == 'invalid_transition'


def test_delete_finish_rolls_back_to_pre_delete_status_on_drain_timeout(monkeypatch) -> None:
    """M3: a real drain wait that never clears rolls the record back to
    whatever status delete found it in (here 'active'), not always
    'failed' -- an active project whose delete timed out on drain is
    still a perfectly usable active project."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())

    from src.services.projects import busy

    # delete_project's own upfront busy check must pass clean; the
    # simulated inflight write appears only once we're already
    # 'deleting' and into delete_project_finish's drain wait -- the
    # exact race M3 exists to catch (a writer picked up between the
    # check and the flip).
    deleting = asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'
    assert deleting.pre_delete_status == 'active'

    from src.services.projects import delete as delete_mod

    monkeypatch.setattr(delete_mod, '_DELETE_DRAIN_TIMEOUT_SECONDS', 0.01)
    monkeypatch.setattr(delete_mod, '_DELETE_DRAIN_POLL_SECONDS', 0.01)
    monkeypatch.setattr(
        busy,
        '_detection_worker_inflight',
        lambda record: [busy.JobRef(kind='detection_worker', job_id='still-busy')],  # noqa: ARG005
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert exc_info.value.detail['error'] == 'project_busy'

    asyncio.run(registry.ensure_fresh())
    rolled_back = registry.get('alpha')
    assert rolled_back is not None
    assert rolled_back.status == 'active'


def test_delete_finish_index_failure_leaves_record_retryable_not_tombstoned() -> None:
    """M4: a failed indices.delete must not be swallowed into a
    tombstone -- the record stays 'deleting' (retryable) and the
    failing index is never actually removed, so a re-issued DELETE can
    retry instead of orphaning it under a slug nothing can reach again."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())
    alpha_record = registry.get('alpha')
    assert alpha_record is not None
    failing_index = sorted(alpha_record.resources.indexes.values())[0]

    real_delete = client.indices.delete

    async def _delete_but_fail_one(*, index: str, ignore=None):
        if index == failing_index:
            raise RuntimeError('simulated cluster hiccup')
        return await real_delete(index=index, ignore=ignore)

    client.indices.delete = _delete_but_fail_one  # type: ignore[method-assign]

    asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert exc_info.value.detail['error'] == 'project_busy'

    asyncio.run(registry.ensure_fresh())
    stuck = registry.get('alpha')
    assert stuck is not None
    assert stuck.status == 'deleting'  # not 'deleted': retryable

    # A retried delete_project_finish, once the transient failure clears,
    # succeeds and tombstones -- idempotent resumability, not a dead end.
    client.indices.delete = real_delete  # type: ignore[method-assign]
    tombstoned = asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert tombstoned.status == 'deleted'


def test_dry_run_index_count_failure_reports_docs_null_not_zero() -> None:
    """m10: an uncountable index reports docs: null, the same rule
    ProjectCounts.validated already follows -- never a made-up 0 that
    looks like "confirmed empty"."""
    client = FakeLifecycleOpenSearch()
    _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))

    async def _boom(*, index, body=None):
        raise RuntimeError('simulated count failure')

    client.count = _boom  # type: ignore[method-assign]

    report = asyncio.run(lifecycle.dry_run_delete(client, slug='alpha'))
    assert report['indexes']
    assert all(entry['docs'] is None for entry in report['indexes'])


def test_drain_timeout_rollback_preserves_a_concurrent_patch(monkeypatch) -> None:
    """m2: delete_project_finish's rollback (on a drain timeout) must be
    built from a FRESH read, not the stale record snapshot it captured
    at its own top -- otherwise a write that lands *during* the (up to
    60s) drain wait is silently discarded even though the write itself
    succeeded. The PATCH here genuinely races the drain wait (via
    asyncio.gather), landing after delete_project_finish's own initial
    read but before its timeout fires."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())

    deleting = asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    from src.services.projects import busy, delete as delete_mod

    monkeypatch.setattr(delete_mod, '_DELETE_DRAIN_TIMEOUT_SECONDS', 0.2)
    monkeypatch.setattr(delete_mod, '_DELETE_DRAIN_POLL_SECONDS', 0.02)
    monkeypatch.setattr(
        busy,
        '_detection_worker_inflight',
        lambda record: [busy.JobRef(kind='detection_worker', job_id='still-busy')],  # noqa: ARG005
    )

    async def _finish_expect_busy():
        try:
            await lifecycle.delete_project_finish(client, slug='alpha')
        except HTTPException as exc:
            return exc
        raise AssertionError('expected delete_project_finish to raise project_busy')

    async def _patch_mid_drain():
        await asyncio.sleep(0.06)  # after finish's initial read, before its timeout
        return await lifecycle.patch_project(
            client,
            slug='alpha',
            display_name='Renamed mid-drain',
            description=None,
            expected_revision=deleting.revision,
        )

    async def _run() -> tuple[Any, Any]:
        return await asyncio.gather(_finish_expect_busy(), _patch_mid_drain())

    finish_result, patch_result = asyncio.run(_run())
    assert isinstance(finish_result, HTTPException)
    assert finish_result.detail['error'] == 'project_busy'
    assert patch_result.display_name == 'Renamed mid-drain'

    asyncio.run(registry.ensure_fresh())
    rolled_back = registry.get('alpha')
    assert rolled_back is not None
    assert rolled_back.status == 'active'
    assert rolled_back.display_name == 'Renamed mid-drain', (
        'm2: the concurrent PATCH must not be silently discarded by the rollback'
    )
