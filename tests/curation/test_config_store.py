"""The config store's per-process snapshot and hot reload over the (in-memory) config repository:
refresh on a moved epoch, staleness, pinned mode, project isolation and the poll loop."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import pytest
from config_store.memory_config import current as memory_config_now

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store import documents
from src.services.config_store.documents import ActiveConflictError
from src.services.config_store.store import ConfigStore, config_store_for


pytestmark = pytest.mark.unbound


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


async def _save(slug: str, kind: str, name: str, **kw: Any) -> Any:
    with bind_project(_record(slug)):
        saved, _ = await documents.save_config(
            kind=kind, name=name, body={}, expected_revision=kw.pop('expected_revision', None)
        )
    return saved


@pytest.mark.asyncio
async def test_a_save_bumps_the_projects_epoch_every_write() -> None:
    memory = memory_config_now()
    pid = memory.project_id('alpha')
    assert memory.epochs.get(pid, 0) == 0
    _, first = await _save_with_epoch('alpha', 'prompt_pack', 'a')
    _, second = await _save_with_epoch('alpha', 'region_profile', 'b')
    assert (first, second) == (1, 2)


async def _save_with_epoch(slug: str, kind: str, name: str) -> tuple[Any, int]:
    with bind_project(_record(slug)):
        return await documents.save_config(kind=kind, name=name, body={}, expected_revision=None)


@pytest.mark.asyncio
async def test_activate_occ_and_rollback() -> None:
    with bind_project(_record('alpha')):
        for name in ('wheel', 'plate'):
            await documents.save_config(
                kind='region_profile', name=name, body={}, expected_revision=None
            )
        result = await documents.activate(
            axis='detection_profile', name='wheel', revision=1, expected_active=None
        )
        assert result.ref == ('wheel', 1)

        with pytest.raises(ActiveConflictError):
            await documents.activate(
                axis='detection_profile',
                name='plate',
                revision=1,
                expected_active=None,  # stale: an activation already exists
            )
        await documents.activate(
            axis='detection_profile',
            name='plate',
            revision=1,
            expected_active={'name': 'wheel', 'revision': 1},
        )
        target = await documents.previous_target('detection_profile')
        assert (target.name, target.revision) == ('wheel', 1)
        rolled_back = await documents.activate(
            axis='detection_profile',
            name=target.name,
            revision=target.revision,
            expected_active={'name': 'plate', 'revision': 1},
            reason='rollback',
        )
    assert rolled_back.ref == ('wheel', 1)


@pytest.mark.asyncio
async def test_two_stores_see_each_others_writes_after_ensure_fresh() -> None:
    """Two ConfigStore instances (two uvicorn workers): B sees A's write after ``ensure_fresh``,
    and not before ``max_age`` has elapsed."""
    store_a = ConfigStore(mode='live', label='alpha')
    store_b = ConfigStore(mode='live', label='alpha')
    await store_a.refresh()
    await store_b.refresh()

    await _save('alpha', 'prompt_pack', 'wheel')

    # B hasn't refreshed yet: a stale snapshot until ensure_fresh forces it.
    assert 'wheel' not in store_b.current.packs
    snapshot = await store_b.ensure_fresh(max_age_s=0.0)
    assert 'wheel' in snapshot.packs
    assert 'wheel' in store_b.current.packs


@pytest.mark.asyncio
async def test_ensure_fresh_skips_refresh_within_max_age() -> None:
    store = ConfigStore(mode='live', label='alpha')
    await store.refresh()
    await _save('alpha', 'prompt_pack', 'wheel')
    # A very generous max_age means the cached (pre-write) snapshot wins.
    snapshot = await store.ensure_fresh(max_age_s=1000.0)
    assert 'wheel' not in snapshot.packs


@pytest.mark.asyncio
async def test_an_unchanged_epoch_reloads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    store = ConfigStore(mode='live', label='alpha')
    await _save('alpha', 'prompt_pack', 'wheel')
    await store.refresh()
    loaded = store.current

    async def boom() -> Any:
        raise AssertionError('the snapshot was reloaded at the same epoch')

    monkeypatch.setattr(store, '_load', boom)
    assert await store.refresh() is loaded


@pytest.mark.asyncio
async def test_refresh_failure_keeps_snapshot_and_sets_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConfigStore(mode='live', label='alpha')
    await _save('alpha', 'prompt_pack', 'wheel')
    await store.refresh()
    assert 'wheel' in store.current.packs
    assert store.current.stale is False

    async def down() -> int:
        raise RuntimeError('database down')

    monkeypatch.setattr(store, '_epoch', down)
    await store.refresh()
    assert store.current.stale is True
    assert 'wheel' in store.current.packs  # last good snapshot kept, not emptied


@pytest.mark.asyncio
async def test_pinned_mode_does_not_publish_until_pin_active() -> None:
    store = ConfigStore(mode='pinned', label='worker-alpha')
    # a pinned worker store reads the project it is labelled with
    await _save('worker-alpha', 'region_profile', 'wheel')
    await store.refresh()
    assert 'wheel' not in store.current.profiles  # staged only
    assert store.pending_snapshot is not None
    store.pin_active()
    assert 'wheel' in store.current.profiles
    assert store.pending_snapshot is None


@pytest.mark.asyncio
async def test_pack_stored_in_one_project_invisible_in_another() -> None:
    """Cross-project isolation: a pack saved in alpha never appears in beta's store."""
    await _save('alpha', 'prompt_pack', 'wheel')
    store_alpha = ConfigStore(mode='live', label='alpha')
    store_beta = ConfigStore(mode='live', label='beta')
    await store_alpha.refresh()
    await store_beta.refresh()
    assert 'wheel' in store_alpha.current.packs
    assert 'wheel' not in store_beta.current.packs
    # Activating in alpha leaves beta's activation untouched.
    with bind_project(_record('alpha')):
        await documents.activate(axis='prompt_pack', name='wheel', revision=1, expected_active=None)
    await store_beta.refresh()
    assert store_beta.current.active_pack is None


def test_get_config_store_isolates_by_bound_project() -> None:
    from src.services.config_store.store import get_config_store, reset_config_stores

    reset_config_stores()
    with bind_project(_record('alpha')):
        store_alpha = get_config_store()
    with bind_project(_record('beta')):
        store_beta = get_config_store()
    assert store_alpha is not store_beta
    assert (store_alpha.label, store_beta.label) == ('alpha', 'beta')


def test_active_config_response_serves_source_activated_at_and_applied() -> None:
    """ActiveConfigResponse (the Cropwright UI reads it)
    must carry `source`, `activated_at` and `applied[]` -- not just
    `axis`/`active`/`previous`/`config_revision`/`stale`."""
    from src.schemas.curation.config_common import ActiveConfigResponse, ActiveRef, AppliedRuntime

    resp = ActiveConfigResponse(
        axis='detection_profile',
        active=ActiveRef(name='vehicle_wheel', revision=3),
        source='stored',
        activated_at='2026-09-26T12:00:00Z',
        previous=ActiveRef(name='generic_item_v1', revision=None),
        config_revision=21,
        stale=False,
        applied=[
            AppliedRuntime(
                process='detection_worker',
                host='opfinal-detection-worker',
                applied_config_revision=21,
                profile=ActiveRef(name='vehicle_wheel', revision=3),
                pack=ActiveRef(name='vehicle_wheel', revision=1),
                vlm=None,
                applied_at='2026-09-26T12:00:05Z',
                lagging=False,
            )
        ],
    )
    payload = resp.model_dump()
    assert payload['source'] == 'stored'
    assert payload['activated_at'] == '2026-09-26T12:00:00Z'
    assert payload['applied'][0]['process'] == 'detection_worker'
    assert payload['applied'][0]['profile'] == {'name': 'vehicle_wheel', 'revision': 3}
    assert payload['applied'][0]['lagging'] is False

    # Defaults: an axis never activated through the store, no worker
    # has ever applied anything.
    env_default = ActiveConfigResponse(
        axis='prompt_pack',
        active=ActiveRef(name='generic_item_v1', revision=None),
        source='env',
        config_revision=0,
    )
    assert env_default.activated_at is None
    assert env_default.applied == []

    off = ActiveConfigResponse(
        axis='detection_profile', active=ActiveRef(), source='off', config_revision=5
    )
    assert off.active.name is None


@pytest.mark.asyncio
async def test_poll_loop_refreshes_every_project_whose_epoch_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The background poll loop fans out over every project whose epoch moved, not just the one
    bound at lifespan startup: an activation made through one uvicorn worker is seen in another
    without some route touching that project's store."""
    from src.services.config_store.store_poll import _poll_all_active_projects

    await _save('alpha', 'region_profile', 'wheel')
    with bind_project(_record('alpha')):
        await documents.activate(
            axis='detection_profile', name='wheel', revision=1, expected_active=None
        )
    memory = memory_config_now()
    memory.project_id('beta')

    async def epochs(known: dict[str, int | None]) -> dict[str, int | None]:
        current: dict[str, int | None] = {
            slug: memory.epochs.get(pid, 0) for pid, slug in memory.slugs.items()
        }
        return {k: v for k, v in current.items() if known.get(k) != v}

    monkeypatch.setattr('src.services.config_store.store_poll.poll_config_epochs', epochs)
    task = asyncio.get_event_loop().create_task(_poll_all_active_projects(0.01))
    try:
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    alpha_store = config_store_for('alpha')
    beta_store = config_store_for('beta')
    # The poll loop, never this test, refreshed alpha's store.
    assert alpha_store.current.active_profile == ('wheel', 1)
    # beta was polled too (its own, unrelated, empty store).
    assert beta_store.current.loaded_at > 0


@pytest.mark.asyncio
async def test_a_poll_whose_refresh_failed_is_retried_on_the_next_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store that marked itself stale stays in the moved set until a refresh succeeds."""
    from src.services.config_store.store_poll import _poll_all_active_projects

    await _save('alpha', 'prompt_pack', 'wheel')
    memory = memory_config_now()
    pid = memory.project_id('alpha')
    calls = {'n': 0}
    real_load = ConfigStore._load

    async def flaky(self: ConfigStore) -> Any:
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('database blip')
        return await real_load(self)

    async def epochs(known: dict[str, int | None]) -> dict[str, int | None]:
        epoch = memory.epochs.get(pid, 0)
        return {} if known.get('alpha') == epoch else {'alpha': epoch}

    monkeypatch.setattr(ConfigStore, '_load', flaky)
    monkeypatch.setattr('src.services.config_store.store_poll.poll_config_epochs', epochs)
    task = asyncio.get_event_loop().create_task(_poll_all_active_projects(0.01))
    try:
        await asyncio.sleep(0.15)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert calls['n'] >= 2
    assert 'wheel' in config_store_for('alpha').current.packs


@pytest.mark.asyncio
async def test_miss_refresh_sees_write_made_through_another_worker() -> None:
    store = ConfigStore(mode='live', label='alpha')
    await store.refresh()
    await _save('alpha', 'region_profile', 'clone1')
    await store.refresh_on_miss()
    assert 'clone1' in store.current.profiles
