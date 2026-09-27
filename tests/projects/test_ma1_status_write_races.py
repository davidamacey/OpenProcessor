"""P3F pass-3 MA1: a status-transition write must never overwrite a
status it never validated. Two independent probes from the review:

1. A slow create (still genuinely in progress past
   ``_BUILDING_STALE_SECONDS``) racing a stale-building delete: the
   create's own final ``active`` write must never resurrect a
   tombstoned slug.
2. A second ``delete_project_finish`` for the SAME slug, triggered
   while a first finish's drain wait is still running, must never be
   able to delete a project that the first finish's own drain-timeout
   rollback has already restored to ``active``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

from src.services.projects import busy, delete as delete_mod, lifecycle
from src.services.projects.registry import ProjectRegistry, doc_to_record, set_project_registry

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


def test_slow_create_cannot_resurrect_a_stale_building_delete(monkeypatch) -> None:
    """Probe 1: block a create inside ``_ensure_indexes`` (standing in
    for a create that has genuinely run past 120s on a degraded
    cluster, not a dead one), let a stale-building DELETE tombstone the
    same slug while it's blocked, then release the create. Before the
    fix, create's final ``active`` write always succeeded (a fresh
    read right before the write trivially matches, so OCC alone never
    catches it) -- the tombstoned slug came back to life."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))

    entered_ensure_indexes = asyncio.Event()
    release_create = asyncio.Event()

    async def _blocking_ensure_indexes(opensearch):
        entered_ensure_indexes.set()
        await release_create.wait()
        await fake_ensure_indexes(opensearch)

    import src.routers.curation._common as common_mod

    monkeypatch.setattr(common_mod, '_ensure_indexes', _blocking_ensure_indexes)

    async def _run() -> None:
        create_task = asyncio.create_task(
            lifecycle.create_project(client, slug='gamma', display_name='Gamma')
        )
        await entered_ensure_indexes.wait()

        # Stand in for "this create has genuinely run past
        # _BUILDING_STALE_SECONDS": push the stored 'building' doc's
        # updated_at far enough into the past for the N1 escape hatch
        # to treat it as stale (the create is NOT actually dead -- it's
        # blocked above, about to resume -- this is exactly the "slow,
        # not dead" scenario the review's probe describes).
        stale_updated_at = (
            datetime.now(UTC) - timedelta(seconds=delete_mod._BUILDING_STALE_SECONDS + 1)
        ).isoformat()
        client.docs['project:gamma']['updated_at'] = stale_updated_at
        # delete_project's own precondition checks (_last_active_check,
        # any registry-snapshot-based read) must also see the doc as
        # stale, not just a raw client.get() -- keep the in-process
        # registry snapshot in sync with the direct mutation above.
        registry._by_slug['gamma'] = doc_to_record(client.docs['project:gamma'])

        deleting = await lifecycle.delete_project(client, slug='gamma', confirm='gamma')
        assert deleting.status == 'deleting'
        tombstoned = await lifecycle.delete_project_finish(client, slug='gamma')
        assert tombstoned.status == 'deleted'

        release_create.set()
        with pytest.raises(HTTPException) as exc_info:
            await create_task
        assert exc_info.value.detail['error'] == 'invalid_transition'

        final_doc = client.docs.get('project:gamma')
        assert final_doc is not None
        assert final_doc['status'] == 'deleted', (
            'MA1: a slow create must never resurrect a tombstoned slug'
        )

    asyncio.run(_run())


def test_second_finish_cannot_delete_a_project_the_first_finish_rolled_back(monkeypatch) -> None:
    """Probe 2: finish A starts draining (never clears -> rolls back to
    'active' on timeout). While A is still draining, finish B is
    triggered for the SAME slug (e.g. the router's M4 retry path after
    a re-DELETE). B must never run to completion: either it is refused
    outright (finish_in_progress) while A still holds the slug, or it
    starts only after A has already finished -- in which case it must
    see A's own rollback (status back to 'active') and refuse the
    transition itself. Either way, the project must never end up
    deleted."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())

    deleting = asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    monkeypatch.setattr(delete_mod, '_DELETE_DRAIN_TIMEOUT_SECONDS', 0.2)
    monkeypatch.setattr(delete_mod, '_DELETE_DRAIN_POLL_SECONDS', 0.02)
    monkeypatch.setattr(
        busy,
        '_detection_worker_inflight',
        lambda record: [busy.JobRef(kind='detection_worker', job_id='still-busy')],  # noqa: ARG005
    )

    async def _finish_a():
        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.delete_project_finish(client, slug='alpha')
        assert exc_info.value.detail['error'] == 'project_busy'

    async def _finish_b_mid_drain():
        await asyncio.sleep(0.1)  # after A started draining, before A's 0.2s timeout
        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.delete_project_finish(client, slug='alpha')
        assert exc_info.value.detail['error'] == 'finish_in_progress'

    async def _run() -> None:
        await asyncio.gather(_finish_a(), _finish_b_mid_drain())

    asyncio.run(_run())

    asyncio.run(registry.ensure_fresh())
    rolled_back = registry.get('alpha')
    assert rolled_back is not None
    assert rolled_back.status == 'active', (
        'MA1: a project A rolled back to active must never be deleted by a racing finish B'
    )
