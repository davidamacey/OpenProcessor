"""W2: the config store's low-level primitives (OCC, revision bump,
per-process snapshot, hot reload) --
docs/design/openprocessor_internal/any_domain_plan.md §3.6/§9 W2."""

from __future__ import annotations

from datetime import UTC, datetime

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
