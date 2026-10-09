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
import time
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
        tombstoned = (await lifecycle.delete_project_finish(client, slug='gamma')).record
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


def test_second_finish_on_a_different_worker_cannot_destroy_a_project_a_first_finish_rolled_back(
    monkeypatch,
) -> None:
    """P3F pass-4 F1 -- the P3 review's pass-3 confirmation found the
    prior pass's ``test_second_finish_cannot_delete_a_project_...`` test
    (above) proves nothing about the ACTUAL production topology: api
    runs ``--workers=32``, and ``delete._FINISH_IN_PROGRESS`` is a
    per-worker-process set. A re-DELETE that lands on a DIFFERENT worker
    (31 times out of 32) has its own, separate, empty copy of that guard
    and cannot see that a finish for this slug is already running
    elsewhere.

    This reproduces the reviewer's exact cross-worker probe: finish B
    gets its own fresh, empty ``_FINISH_IN_PROGRESS`` set (simulating a
    second worker process) instead of sharing A's. Busy is time-gated to
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

    deleting = asyncio.run(lifecycle.delete_project(client, slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    monkeypatch.setattr(delete_mod, '_DELETE_DRAIN_TIMEOUT_SECONDS', 0.2)
    monkeypatch.setattr(delete_mod, '_DELETE_DRAIN_POLL_SECONDS', 0.02)

    t0 = time.monotonic()

    def _inflight(record):
        # Busy until just after A's 0.2s drain deadline, then clears --
        # this is what lets A time out and roll back, and lets B's own
        # (later-starting) drain clear shortly after, reaching the real
        # destructive interleaving the review's probe found.
        return (
            [busy.JobRef(kind='detection_worker', job_id='still-busy')]
            if time.monotonic() - t0 < 0.24
            else []
        )

    monkeypatch.setattr(busy, '_detection_worker_inflight', _inflight)

    async def _finish_a() -> None:
        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.delete_project_finish(client, slug='alpha')
        assert exc_info.value.detail['error'] == 'project_busy'

    async def _finish_b_cross_worker() -> None:
        await asyncio.sleep(0.1)  # after A started draining, before A's 0.2s deadline
        # Simulate finish B running on a DIFFERENT worker process: strip
        # A's entry from the (per-process, real) guard so B's check sees
        # an empty state for this slug, exactly like a second worker's
        # own separate, empty `_FINISH_IN_PROGRESS` set would. Mutating
        # the existing set in place (not swapping in a whole new set
        # object) keeps A's own `finally: _FINISH_IN_PROGRESS.discard(...)`
        # -- which resolves the module-global name at call time -- acting
        # on the SAME object B just mutated, so both coroutines' cleanup
        # still converges on one consistent, empty set once both are
        # done (no leaked 'alpha' entry poisoning a later, unrelated
        # test).
        delete_mod._FINISH_IN_PROGRESS.discard('alpha')
        with pytest.raises(HTTPException) as exc_info:
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
    ``_ensure_indexes``. When the create resumes, it creates gamma's
    indexes, then loses the MA1 ``expect_status='building'`` race on its
    final ``active`` write (the fresh status is now ``'deleted'``) and
    correctly refuses to resurrect the slug -- but by then the indexes it
    just made are unreachable forever under a retired slug. This asserts
    the F2 best-effort cleanup actually runs: every one of gamma's
    indexes ends up in ``client.deleted_indexes``, and none remain in
    ``client.indexes``."""
    from src.config.curation import base_curation_config

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

        stale_updated_at = (
            datetime.now(UTC) - timedelta(seconds=delete_mod._BUILDING_STALE_SECONDS + 1)
        ).isoformat()
        client.docs['project:gamma']['updated_at'] = stale_updated_at
        registry._by_slug['gamma'] = doc_to_record(client.docs['project:gamma'])

        deleting = await lifecycle.delete_project(client, slug='gamma', confirm='gamma')
        assert deleting.status == 'deleting'
        tombstoned = (await lifecycle.delete_project_finish(client, slug='gamma')).record
        assert tombstoned.status == 'deleted'

        release_create.set()
        with pytest.raises(HTTPException) as exc_info:
            await create_task
        assert exc_info.value.detail['error'] == 'invalid_transition'

    asyncio.run(_run())

    from src.config.projects import resources_for_new

    gamma_indexes = sorted(set(resources_for_new('gamma', base_curation_config()).indexes.values()))
    assert gamma_indexes, 'sanity: gamma must own at least one index'
    for name in gamma_indexes:
        assert name in client.deleted_indexes, (
            f'F2: {name} was orphaned by the resurrection race and must be best-effort cleaned up'
        )
        assert name not in client.indexes, f'F2: {name} must not still exist after cleanup'
