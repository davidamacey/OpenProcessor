"""P3F pass-3 MA1: a status-transition write must never overwrite a
status it never validated. Two independent probes from the review:

1. A slow create (still genuinely in progress past
   ``_BUILDING_STALE_SECONDS``) racing a stale-building delete: the
   create's own final ``active`` write must never resurrect a
   tombstoned slug.
2. A second ``delete_project_finish`` for the SAME slug (another
   process; the job's exclusive scope normally keeps it from starting),
   triggered while a first finish's drain wait is still running, must
   never be able to delete a project that the first finish's own
   drain-timeout rollback has already restored to ``active``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from src.core.exceptions import RequestRejectedError
from src.services.projects import busy, delete as delete_mod, lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import (
    FakeLifecycleOpenSearch,
    amutate_project,
    fake_ensure_indexes,
    seed_default_project,
)


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


def _registry_for(client: FakeLifecycleOpenSearch) -> ProjectRegistry:
    registry = ProjectRegistry()
    set_project_registry(registry)
    return registry


def test_slow_create_cannot_resurrect_a_stale_building_delete(monkeypatch) -> None:
    """Probe 1: block a create inside ``ensure_indexes`` (standing in
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

    import src.services.curation.index_bootstrap as bootstrap_mod

    monkeypatch.setattr(bootstrap_mod, 'ensure_indexes', _blocking_ensure_indexes)

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
        stale_updated_at = datetime.now(UTC) - timedelta(
            seconds=delete_mod._BUILDING_STALE_SECONDS + 1
        )
        await amutate_project('gamma', updated_at=stale_updated_at)
        # delete_project's own precondition checks (_last_active_check,
        # any registry-snapshot-based read) must also see the doc as
        # stale, not just a raw client.get() -- keep the in-process
        # registry snapshot in sync with the direct mutation above.
        await registry.refresh_strict()

        deleting = await lifecycle.delete_project(slug='gamma', confirm='gamma')
        assert deleting.status == 'deleting'
        tombstoned = (await lifecycle.delete_project_finish(client, slug='gamma')).record
        assert tombstoned.status == 'deleted'

        release_create.set()
        with pytest.raises(RequestRejectedError) as exc_info:
            await create_task
        assert exc_info.value.detail['error'] == 'project_not_found'

        await registry.refresh_strict()
        assert registry.get('gamma') is None, (
            'MA1: a slow create must never resurrect a tombstoned slug'
        )
        assert 'gamma' in registry.retired_slugs()

    asyncio.run(_run())


def test_second_finish_on_a_different_worker_cannot_destroy_a_project_a_first_finish_rolled_back(
    monkeypatch,
) -> None:
    """P3F pass-4 F1 -- a second finish for one slug in another process
    (the exclusive job scope normally prevents it; a reaper resume or a
    manual call may not) shares no in-process state with the first.

    Finish B runs while finish A is draining. Busy is time-gated to
    clear shortly AFTER A's drain deadline (not held forever -- nit n-g:
    a busy check that never clears makes B time out too, and the project
    is never actually destroyed either way, which proves nothing). That
    lets A time out and roll the record back to 'active', and then lets
    B's own drain clear a little later, reaching the point where --
    without the F1 claim write -- B went on to unload models and delete
    every one of alpha's indexes, refusing only at the final tombstone
    write (by then far too late: an 'active' project record whose data is
    already gone). With the F1 claim write, B must stop at that claim,
    before touching a single index."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())

    deleting = asyncio.run(lifecycle.delete_project(slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    # The first finish to start draining is A: it times out (and rolls back); B's drain outlasts
    # it, so the order does not depend on how fast either one reaches the drain.
    drains: list[int] = []
    real_drain = delete_mod._wait_for_drain

    async def _drain(record):
        monkeypatch.setattr(
            delete_mod, '_DELETE_DRAIN_TIMEOUT_SECONDS', 0.4 if not drains else 10.0
        )
        drains.append(1)
        return await real_drain(record)

    monkeypatch.setattr(delete_mod, '_wait_for_drain', _drain)
    monkeypatch.setattr(delete_mod, '_DELETE_DRAIN_POLL_SECONDS', 0.02)

    from src.db.repos import projects as repo

    rolled_back_by_a = {'done': False}
    real_rollback = repo.rollback_delete

    async def _rollback(session, slug):
        row = await real_rollback(session, slug)
        rolled_back_by_a['done'] = True
        return row

    monkeypatch.setattr(repo, 'rollback_delete', _rollback)

    def _inflight(record):
        # Busy until A has timed out and rolled the project back, then clear -- this lets B's own
        # later drain clear, reaching the real destructive interleaving the review's probe found.
        # Driven by the rollback itself, not the clock, so a slow database cannot reorder it.
        return (
            []
            if rolled_back_by_a['done']
            else [busy.JobRef(kind='detection_worker', job_id='still-busy')]
        )

    monkeypatch.setattr(busy, '_detection_worker_inflight', _inflight)

    async def _finish_a() -> None:
        with pytest.raises(RequestRejectedError) as exc_info:
            await lifecycle.delete_project_finish(client, slug='alpha')
        assert exc_info.value.detail['error'] == 'project_busy'

    async def _finish_b_cross_worker() -> None:
        while not drains:  # A is draining
            await asyncio.sleep(0.005)
        with pytest.raises(RequestRejectedError) as exc_info:
            await lifecycle.delete_project_finish(client, slug='alpha')
        assert exc_info.value.detail['error'] == 'invalid_transition', (
            'F1: the claim write must be what stops B, once A has already rolled the '
            'record back -- not a swallowed exception further down the finish'
        )

    async def _run() -> None:
        await asyncio.gather(_finish_a(), _finish_b_cross_worker())

    asyncio.run(_run())

    asyncio.run(registry.ensure_fresh())
    final = registry.get('alpha')
    assert final is not None
    assert final.status == 'active', (
        'F1: a project A already rolled back to active must never be destroyed by a '
        "cross-worker finish B that never shared A's in-process guard"
    )
    assert client.deleted_indexes == [], (
        'F1: B must be stopped by its own claim write BEFORE deleting a single index -- '
        'a finish that only refuses at the final tombstone write is too late'
    )


def test_create_best_effort_cleans_up_indexes_orphaned_by_a_stale_building_delete(
    monkeypatch,
) -> None:
    """P3F pass-4 F2 (documented known gap, best-effort cleanup): same
    race as ``test_slow_create_cannot_resurrect_a_stale_building_delete``
    (probe 1) -- a stale-``building`` delete tombstones ``gamma`` while
    its own (still genuinely running, just slow) create is blocked inside
    ``ensure_indexes``. When the create resumes, it creates gamma's
    indexes, then loses the MA1 ``expect_status='building'`` race on its
    final ``active`` write (the fresh status is now ``'deleted'``) and
    correctly refuses to resurrect the slug -- but by then the indexes it
    just made are unreachable forever under a retired slug. This asserts
    the F2 best-effort cleanup actually runs: every one of gamma's
    indexes ends up in ``client.deleted_indexes``, and none remain in
    ``client.indexes``."""
    from src.config.curation_accessors import base_curation_config

    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))

    entered_ensure_indexes = asyncio.Event()
    release_create = asyncio.Event()

    async def _blocking_ensure_indexes(opensearch):
        entered_ensure_indexes.set()
        await release_create.wait()
        await fake_ensure_indexes(opensearch)

    import src.services.curation.index_bootstrap as bootstrap_mod

    monkeypatch.setattr(bootstrap_mod, 'ensure_indexes', _blocking_ensure_indexes)

    async def _run() -> None:
        create_task = asyncio.create_task(
            lifecycle.create_project(client, slug='gamma', display_name='Gamma')
        )
        await entered_ensure_indexes.wait()

        stale_updated_at = datetime.now(UTC) - timedelta(
            seconds=delete_mod._BUILDING_STALE_SECONDS + 1
        )
        await amutate_project('gamma', updated_at=stale_updated_at)
        await registry.refresh_strict()

        deleting = await lifecycle.delete_project(slug='gamma', confirm='gamma')
        assert deleting.status == 'deleting'
        tombstoned = (await lifecycle.delete_project_finish(client, slug='gamma')).record
        assert tombstoned.status == 'deleted'

        release_create.set()
        with pytest.raises(RequestRejectedError) as exc_info:
            await create_task
        assert exc_info.value.detail['error'] == 'project_not_found'

    asyncio.run(_run())

    from src.config.projects import resources_for_new

    gamma_indexes = sorted(set(resources_for_new('gamma', base_curation_config()).indexes.values()))
    assert gamma_indexes, 'sanity: gamma must own at least one index'
    for name in gamma_indexes:
        assert name in client.deleted_indexes, (
            f'F2: {name} was orphaned by the resurrection race and must be best-effort cleaned up'
        )
        assert name not in client.indexes, f'F2: {name} must not still exist after cleanup'
