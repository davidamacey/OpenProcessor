"""W2: the config store's low-level primitives (OCC, revision bump,
per-process snapshot, hot reload) --
docs/design/openprocessor_internal/any_domain_plan.md §3.6/§9 W2."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store.index import (
    ActiveConflictError,
    RevisionConflictError,
    activate,
    delete_config,
    get_activation,
    get_config_revision,
    save_config,
)
from src.services.config_store.store import ConfigStore


pytestmark = pytest.mark.unbound

INDEX = 'op_prj_alpha__configs'


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


@pytest.mark.asyncio
async def test_save_config_occ_conflict() -> None:
    client = FakeConfigOpenSearch()
    doc = await save_config(
        client, INDEX, kind='prompt_pack', name='wheel', body={'a': 1}, expected_revision=None
    )
    assert doc['revision'] == 1
    # A stale expected_revision (0, as if the caller never saw rev 1) conflicts.
    with pytest.raises(RevisionConflictError) as exc_info:
        await save_config(
            client, INDEX, kind='prompt_pack', name='wheel', body={'a': 2}, expected_revision=0
        )
    assert exc_info.value.current_revision == 1

    # The correct expected_revision succeeds and bumps to 2.
    doc2 = await save_config(
        client, INDEX, kind='prompt_pack', name='wheel', body={'a': 2}, expected_revision=1
    )
    assert doc2['revision'] == 2


@pytest.mark.asyncio
async def test_save_config_bumps_revision_every_write() -> None:
    client = FakeConfigOpenSearch()
    assert await get_config_revision(client, INDEX) == 0
    await save_config(client, INDEX, kind='prompt_pack', name='a', body={}, expected_revision=None)
    assert await get_config_revision(client, INDEX) == 1
    await save_config(
        client, INDEX, kind='region_profile', name='b', body={}, expected_revision=None
    )
    assert await get_config_revision(client, INDEX) == 2


@pytest.mark.asyncio
async def test_delete_then_recreate_never_reuses_revision() -> None:
    client = FakeConfigOpenSearch()
    await save_config(
        client, INDEX, kind='prompt_pack', name='wheel', body={}, expected_revision=None
    )
    doc = await save_config(
        client, INDEX, kind='prompt_pack', name='wheel', body={}, expected_revision=1
    )
    assert doc['revision'] == 2
    await delete_config(client, INDEX, kind='prompt_pack', name='wheel', expected_revision=2)
    recreated = await save_config(
        client, INDEX, kind='prompt_pack', name='wheel', body={}, expected_revision=None
    )
    assert recreated['revision'] == 3  # continues past the highest revision copy (2), never 1


@pytest.mark.asyncio
async def test_activate_occ_and_rollback() -> None:
    client = FakeConfigOpenSearch()
    result = await activate(
        client, INDEX, axis='detection_profile', name='wheel', revision=None, expected_active=None
    )
    assert result['name'] == 'wheel'
    current = await get_activation(client, INDEX, 'detection_profile')
    assert current is not None
    assert current['name'] == 'wheel'

    with pytest.raises(ActiveConflictError):
        await activate(
            client,
            INDEX,
            axis='detection_profile',
            name='other',
            revision=None,
            expected_active=None,  # stale: activation already exists
        )

    from src.services.config_store.index import rollback

    await activate(
        client,
        INDEX,
        axis='detection_profile',
        name='plate',
        revision=None,
        expected_active={'name': 'wheel', 'revision': None},
    )
    rolled_back = await rollback(
        client, INDEX, axis='detection_profile', expected_active={'name': 'plate', 'revision': None}
    )
    assert rolled_back['name'] == 'wheel'


@pytest.mark.asyncio
async def test_two_stores_see_each_others_writes_after_ensure_fresh() -> None:
    """Two ConfigStore instances on one fake OpenSearch (simulating two
    uvicorn workers): B sees A's write after ``ensure_fresh``, and not
    before ``max_age`` has elapsed."""
    client = FakeConfigOpenSearch()
    store_a = ConfigStore(index=INDEX, mode='live', label='a')
    store_b = ConfigStore(index=INDEX, mode='live', label='b')
    await store_a.refresh(client)
    await store_b.refresh(client)

    await save_config(
        client, INDEX, kind='prompt_pack', name='wheel', body={}, expected_revision=None
    )
    store_a.apply_local(config_revision=1)  # writer applies its own write locally

    # B hasn't refreshed yet -- stale snapshot until ensure_fresh forces it.
    assert 'wheel' not in store_b.current.packs
    snapshot = await store_b.ensure_fresh(client, max_age_s=0.0)
    assert 'wheel' in snapshot.packs
    assert 'wheel' in store_b.current.packs


@pytest.mark.asyncio
async def test_ensure_fresh_skips_refresh_within_max_age() -> None:
    client = FakeConfigOpenSearch()
    store = ConfigStore(index=INDEX, mode='live', label='a')
    await store.refresh(client)
    await save_config(
        client, INDEX, kind='prompt_pack', name='wheel', body={}, expected_revision=None
    )
    # A very generous max_age means the cached (pre-write) snapshot wins.
    snapshot = await store.ensure_fresh(client, max_age_s=1000.0)
    assert 'wheel' not in snapshot.packs


@pytest.mark.asyncio
async def test_refresh_failure_keeps_snapshot_and_sets_stale() -> None:
    client = FakeConfigOpenSearch()
    store = ConfigStore(index=INDEX, mode='live', label='a')
    await save_config(
        client, INDEX, kind='prompt_pack', name='wheel', body={}, expected_revision=None
    )
    await store.refresh(client)
    assert 'wheel' in store.current.packs
    assert store.current.stale is False

    class _BrokenClient:
        async def get(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError('opensearch down')

    await store.refresh(_BrokenClient())
    assert store.current.stale is True
    assert 'wheel' in store.current.packs  # last good snapshot kept, not emptied


@pytest.mark.asyncio
async def test_pinned_mode_does_not_publish_until_pin_active() -> None:
    client = FakeConfigOpenSearch()
    store = ConfigStore(index=INDEX, mode='pinned', label='worker')
    await save_config(
        client, INDEX, kind='region_profile', name='wheel', body={}, expected_revision=None
    )
    await store.refresh(client)
    assert 'wheel' not in store.current.profiles  # staged only
    assert store.pending_snapshot is not None
    store.pin_active()
    assert 'wheel' in store.current.profiles
    assert store.pending_snapshot is None


@pytest.mark.asyncio
async def test_pack_stored_in_one_project_invisible_in_another() -> None:
    """Cross-project isolation: a pack saved against alpha's configs
    index never appears in beta's ConfigStore, even sharing one fake
    OpenSearch client."""
    client = FakeConfigOpenSearch()
    alpha_index = 'op_prj_alpha__configs'
    beta_index = 'op_prj_beta__configs'
    await save_config(
        client, alpha_index, kind='prompt_pack', name='wheel', body={}, expected_revision=None
    )
    store_alpha = ConfigStore(index=alpha_index, mode='live', label='alpha')
    store_beta = ConfigStore(index=beta_index, mode='live', label='beta')
    await store_alpha.refresh(client)
    await store_beta.refresh(client)
    assert 'wheel' in store_alpha.current.packs
    assert 'wheel' not in store_beta.current.packs
    # Activating in alpha leaves beta's activation untouched.
    await activate(
        client, alpha_index, axis='prompt_pack', name='wheel', revision=1, expected_active=None
    )
    await store_beta.refresh(client)
    assert store_beta.current.active_pack is None


def test_get_config_store_isolates_by_bound_project() -> None:
    from src.services.config_store.store import get_config_store, reset_config_stores

    reset_config_stores()
    with bind_project(_record('alpha')):
        store_alpha = get_config_store()
    with bind_project(_record('beta')):
        store_beta = get_config_store()
    assert store_alpha is not store_beta
    assert store_alpha.index != store_beta.index
    assert store_alpha.index == 'op_prj_alpha__configs'
    assert store_beta.index == 'op_prj_beta__configs'


@pytest.mark.asyncio
async def test_refresh_forces_the_index_current_before_searching_configs() -> None:
    """B5 (reviewer probe #8): a save+activate followed by a refresh
    must not cache a snapshot whose packs/profiles disagree with the
    just-bumped revision. Reproduced against a fake whose ``search`` lags
    a real near-real-time index until ``indices.refresh()`` runs --
    ``ConfigStore._load_snapshot`` must issue that refresh itself before
    reading configs, not rely on the caller."""
    from curation._fake_config_opensearch import NearRealTimeConfigOpenSearch

    client = NearRealTimeConfigOpenSearch()
    store = ConfigStore(index=INDEX, mode='live')

    doc = await save_config(
        client, INDEX, kind='prompt_pack', name='mypack', body={'x': 1}, expected_revision=None
    )
    await activate(
        client,
        INDEX,
        axis='prompt_pack',
        name='mypack',
        revision=doc['revision'],
        expected_active=None,
    )

    snapshot = await store.refresh(client)
    assert snapshot.active_pack == ('mypack', doc['revision'])
    assert 'mypack' in snapshot.packs, (
        f'snapshot paired revision {snapshot.config_revision} with a stale doc set '
        f'(packs={list(snapshot.packs)}) -- _load_snapshot did not force the index '
        'current before searching'
    )


def test_active_config_response_serves_source_activated_at_and_applied() -> None:
    """Cropwright W3 UI (C2/Q5, any_domain_plan.md §7.2): ActiveConfigResponse
    must carry `source`, `activated_at` and `applied[]` -- not just
    `axis`/`active`/`previous`/`config_revision`/`stale`."""
    from src.routers.curation._config_common_models import (
        ActiveConfigResponse,
        ActiveRef,
        AppliedRuntime,
    )

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
async def test_poll_all_active_projects_refreshes_every_project_not_just_one() -> None:
    """M4: the background poll loop must fan out over every active
    project's own store, not just whichever project happened to be
    bound at lifespan startup -- otherwise an activation made through
    one uvicorn worker stays invisible in another until some other
    route happens to touch that project's store."""
    from curation._fake_config_opensearch import TwoProjectOpenSearch
    from src.config.curation import base_curation_config
    from src.config.projects import resources_for_new
    from src.services.config_store.store import _poll_all_active_projects, get_config_store
    from src.services.projects.registry import ProjectRecord, record_to_doc

    client = TwoProjectOpenSearch()
    projects_index = 'op_projects'
    client._docs.setdefault(projects_index, {})

    def _doc(slug: str) -> dict[str, Any]:
        return record_to_doc(
            ProjectRecord(
                slug=slug,
                display_name=slug,
                description='',
                status='active',
                revision=1,
                created_at='',
                updated_at='',
                origin=None,
                resources=resources_for_new(slug, base_curation_config()),
            )
        )

    client._docs[projects_index]['project:alpha'] = {'_source': _doc('alpha'), '_seq_no': 0}
    client._docs[projects_index]['project:beta'] = {'_source': _doc('beta'), '_seq_no': 0}

    with bind_project(_record('alpha')):
        idx = get_config_store(mode='live').index
    doc = await save_config(
        client, idx, kind='region_profile', name='wheel', body={}, expected_revision=None
    )
    await activate(
        client,
        idx,
        axis='detection_profile',
        name='wheel',
        revision=doc['revision'],
        expected_active=None,
    )

    poll_task = asyncio.get_event_loop().create_task(_poll_all_active_projects(client, 0.01))
    try:
        await asyncio.sleep(0.05)
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task

    with bind_project(_record('alpha')):
        alpha_store = get_config_store(mode='live')
    with bind_project(_record('beta')):
        beta_store = get_config_store(mode='live')

    # The poll loop, never this test, refreshed alpha's store.
    assert alpha_store.current.active_profile == ('wheel', doc['revision'])
    # beta was polled too (its own, unrelated, empty store).
    assert beta_store.current.loaded_at > 0


@pytest.mark.asyncio
async def test_startup_bootstrap_config_store_safe_starts_poll_task_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MJ1 (W2-finish review, 2026-09-27): ``src.main``'s lifespan runs
    unbound -- no project is EVER bound at process startup -- so the old
    code's ``get_config_store(mode='live')`` + ``store.refresh(client)``
    "warm the bound project's store" step always raised
    ``ProjectNotBound``. The function's own broad ``except`` swallowed
    that and returned ``None``, so the poll task never started in any
    real deployment. Reproduces the reviewer's exact probe: call the
    real function with no project bound, a fake client, and assert it
    returns a task whose first tick refreshes an active project's store."""
    from curation._fake_config_opensearch import TwoProjectOpenSearch
    from src.services.config_store.store import (
        get_config_store,
        reset_config_stores,
        reset_global_config_store,
        shutdown_config_store_poll,
        startup_bootstrap_config_store_safe,
    )
    from src.services.projects import guard
    from src.services.projects.registry import record_to_doc

    reset_config_stores()
    reset_global_config_store()

    client = TwoProjectOpenSearch()
    projects_index = 'op_projects'
    client._docs.setdefault(projects_index, {})
    client._docs[projects_index]['project:alpha'] = {
        '_source': record_to_doc(_record('alpha')),
        '_seq_no': 0,
    }

    async def _fake_make_curation_opensearch() -> Any:
        return client

    monkeypatch.setattr(guard, 'make_curation_opensearch', _fake_make_curation_opensearch)
    monkeypatch.setenv('OP_CONFIG_POLL_S', '0.01')

    task = await startup_bootstrap_config_store_safe()
    assert task is not None  # MJ1: used to be None -- no poll task at all
    try:
        for _ in range(200):
            with bind_project(_record('alpha')):
                alpha_store = get_config_store(mode='live')
            if alpha_store.current.loaded_at > 0:
                break
            await asyncio.sleep(0.01)
        with bind_project(_record('alpha')):
            alpha_store = get_config_store(mode='live')
        assert alpha_store.current.loaded_at > 0
    finally:
        await shutdown_config_store_poll(task)
        reset_config_stores()
        reset_global_config_store()
