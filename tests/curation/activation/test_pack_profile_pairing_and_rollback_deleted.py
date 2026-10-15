"""Pack and profile pairing in combined requests and clone, and rollback to a deleted
revision."""

from __future__ import annotations

import asyncio
import contextlib
import tempfile

import pytest

from curation.activation.constants import (
    PREFIX,
    RP,
    SETTINGS,
    live_pair,
    pack_body,
    stripped_pack_body,
)


pytestmark = pytest.mark.usefixtures('vlm_env_active')


def _setup_packs(c):
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body('GOOD')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'single', 'body': stripped_pack_body()}).status_code == 201


@pytest.mark.parametrize('order', ['pack_first', 'profile_first'])
def test_combined_put_accepts_pair_valid_only_as_a_pair(
    app_client,
    ready_state,
    registry,
    order,
):
    """single+base live. {good, multi} is a VALID new pairing even though
    multi is invalid against the OLD pack. Must 200."""
    c = app_client
    _setup_packs(c)
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single'}}).status_code == 200
    d = (
        {'prompt_pack': 'good', 'detection_profile': 'multi'}
        if order == 'pack_first'
        else {'detection_profile': 'multi', 'prompt_pack': 'good'}
    )
    r = c.put(SETTINGS, json={'defaults': d})
    pack, prof = live_pair()
    print(order, r.status_code, r.text[:200], pack.class_system, prof and prof.name)
    assert r.status_code == 200
    assert pack.class_system == 'GOOD'
    assert prof is not None
    assert prof.name == 'multi'


def test_combined_put_accepts_stripped_pack_with_profile_off(app_client, ready_state, registry):
    c = app_client
    _setup_packs(c)
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}}).status_code == 200
    r = c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single', 'detection_profile': 'off'}})
    pack, prof = live_pair()
    print(r.status_code, r.text[:200], pack.class_system, prof)
    assert r.status_code == 200
    assert pack.class_system == 'STRIPPED'
    assert prof is None


def test_combined_put_pack_off_with_multi_is_validated_against_env_default(
    app_client,
    ready_state,
    registry,
):
    """Pack -> None (env default GENERIC pack, has multi-box keys) + multi."""
    c = app_client
    _setup_packs(c)
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single'}}).status_code == 200
    r = c.put(SETTINGS, json={'defaults': {'prompt_pack': None, 'detection_profile': 'multi'}})
    pack, prof = live_pair()
    print(r.status_code, r.text[:200], pack.name, pack.class_system, prof and prof.name)
    assert r.status_code == 200
    assert pack.class_system != 'STRIPPED'


def test_combined_put_rejects_stripped_pack_under_live_multi_profile(
    app_client, ready_state, registry
):
    c = app_client
    _setup_packs(c)
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}}).status_code == 200
    r = c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single', 'detection_profile': 'multi'}})
    pack, _prof = live_pair()
    print(r.status_code, r.text[:200], pack.class_system)
    assert r.status_code == 422
    assert pack.class_system == 'GOOD'


def test_rollback_to_deleted_pack_is_not_resurrected(app_client):
    """A pack deleted while it is the activation's `previous`: its @rev copy is
    still found, but rollback must not resurrect a pack that no longer exists in the
    list."""
    from src.services.labeling.vlm_prompts import active_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': pack_body('A1')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'pack_b', 'body': pack_body('B1')}).status_code == 201
    assert c.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.post(
            f'{PREFIX}/pack_b/activate', json={'expected_active': {'name': 'pack_a', 'revision': 1}}
        ).status_code
        == 200
    )
    d = c.request('DELETE', f'{PREFIX}/pack_a', json={'expected_revision': 1})
    if d.status_code != 204:
        d = c.delete(f'{PREFIX}/pack_a?expected_revision=1')
    rb = c.post(
        f'{PREFIX}/active/rollback', json={'expected_active': {'name': 'pack_b', 'revision': 1}}
    )
    listed = (
        [p['name'] for p in c.get(PREFIX).json().get('packs', c.get(PREFIX).json())]
        if isinstance(c.get(PREFIX).json(), (dict, list))
        else None
    )
    act = c.get(f'{PREFIX}/active').json()
    print(
        'delete',
        d.status_code,
        '| rollback',
        rb.status_code,
        rb.text[:160],
        '| active',
        act.get('active'),
        '| served',
        active_prompt_pack().class_system,
        '| listed',
        listed,
    )
    # Correct: refuse to make a deleted pack active (404/409), not resurrect it.
    assert not (rb.status_code == 200 and act['active']['name'] == 'pack_a')


def test_rollback_to_deleted_pack_is_409_previous_deleted(app_client):
    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': pack_body('A1')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'pack_b', 'body': pack_body('B1')}).status_code == 201
    assert c.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.post(
            f'{PREFIX}/pack_b/activate', json={'expected_active': {'name': 'pack_a', 'revision': 1}}
        ).status_code
        == 200
    )
    d = c.delete(f'{PREFIX}/pack_a?expected_revision=1')
    if d.status_code not in (200, 204):
        d = c.request('DELETE', f'{PREFIX}/pack_a', json={'expected_revision': 1})
    assert d.status_code in (200, 204), d.text
    rb = c.post(
        f'{PREFIX}/active/rollback', json={'expected_active': {'name': 'pack_b', 'revision': 1}}
    )
    act = c.get(f'{PREFIX}/active').json()
    print(rb.status_code, rb.text[:200], act)
    assert rb.status_code == 409
    assert 'previous_deleted' in rb.text
    assert act['active']['name'] == 'pack_b'
    assert act['source'] == 'stored'


def test_rollback_to_deleted_profile_is_409_previous_deleted(app_client, ready_state, registry):
    c = app_client
    prof = {'detector_model': 'wheel_detector', 'text_reader': 'none'}
    for n in ('rp_a', 'rp_b'):
        r = c.post(RP, json={'name': n, 'body': prof})
        assert r.status_code == 201, r.text
    assert c.post(f'{RP}/rp_a/activate', json={'expected_active': None}).status_code == 200
    r = c.post(f'{RP}/rp_b/activate', json={'expected_active': {'name': 'rp_a', 'revision': 1}})
    assert r.status_code == 200, r.text
    d = c.delete(f'{RP}/rp_a?expected_revision=1')
    if d.status_code not in (200, 204):
        d = c.request('DELETE', f'{RP}/rp_a', json={'expected_revision': 1})
    assert d.status_code in (200, 204), d.text
    rb = c.post(f'{RP}/active/rollback', json={'expected_active': {'name': 'rp_b', 'revision': 1}})
    print(rb.status_code, rb.text[:200])
    assert rb.status_code == 409
    assert 'previous_deleted' in rb.text


def test_rollback_refreshes_stale_snapshot_before_gating(app_client, ready_state, registry):
    """Another process activates 'multi' (direct index write, this
    process's snapshot is stale and older than the TTL). Rolling back to
    the stripped pack must see 'multi' and 422."""
    from src.config import get_curation_config
    from src.services.config_store import get_config_store
    from src.services.config_store.index import activate

    c = app_client
    _setup_packs(c)
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single'}}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}}).status_code == 200
    idx = get_curation_config().configs_index
    store = get_config_store()
    cur = store.current.active_profile
    exp = None if cur is None else ({'name': None, 'revision': None} if cur == 'off' else None)
    asyncio.run(
        activate(
            c.fake_os,
            idx,
            axis='detection_profile',
            name='multi',
            revision=None,
            expected_active=exp,
        )
    )
    # Age the snapshot past the 1 s TTL without refreshing it.
    from dataclasses import replace

    store.current = replace(store.current, loaded_at=store.current.loaded_at - 10)
    rb = c.post(
        f'{PREFIX}/active/rollback', json={'expected_active': {'name': 'good', 'revision': 1}}
    )
    pack, prof = live_pair()
    print(rb.status_code, rb.text[:200], pack.class_system, prof and prof.name)
    assert rb.status_code == 422
    assert pack.class_system == 'GOOD'


def _clone_pair(profile_activation: tuple[str | None, int | None] | None, plant_profile: bool):
    """Plant pack 'single' (STRIPPED) active in the source plus the given
    profile activation, clone into a fresh target, return the target's
    effective live (pack, profile)."""
    import os
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, MagicMock

    import src.config.curation_accessors as curation_mod
    from curation._fake_config_opensearch import FakeConfigOpenSearch
    from src.config.curation_accessors import base_curation_config
    from src.config.project_context import bind_project
    from src.config.projects import ProjectRecord, resources_for_new
    from src.services.config_store import get_config_store
    from src.services.config_store.index import activate, save_config
    from src.services.config_store.store import reset_config_stores

    class _Fake(FakeConfigOpenSearch):
        async def count(self, index, body=None):  # noqa: ARG002
            return {'count': 0}

        async def bulk(self, body, refresh=False):  # noqa: ARG002
            return {'errors': False, 'items': []}

    def _rec(slug):
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

    with tempfile.TemporaryDirectory() as tmp:
        old = {k: os.environ.get(k) for k in ('OP_STATE_DIR', 'OP_PROJECTS_DATA_ROOT')}
        os.environ['OP_STATE_DIR'] = f'{tmp}/state'
        os.environ['OP_PROJECTS_DATA_ROOT'] = f'{tmp}/pd'
        curation_mod._default_curation_config = None
        reset_config_stores()
        try:
            client = _Fake()
            source, target = _rec('alpha'), _rec('beta')
            with bind_project(source):
                from src.config import get_curation_config

                idx = get_curation_config().configs_index
                asyncio.run(
                    save_config(
                        client,
                        idx,
                        kind='prompt_pack',
                        name='single',
                        body=stripped_pack_body(),
                        expected_revision=None,
                    )
                )
                if profile_activation is not None:
                    pname, prev = profile_activation
                    asyncio.run(
                        activate(
                            client,
                            idx,
                            axis='detection_profile',
                            name=pname,
                            revision=prev,
                            expected_active=None,
                        )
                    )
                asyncio.run(
                    activate(
                        client,
                        idx,
                        axis='prompt_pack',
                        name='single',
                        revision=1,
                        expected_active=None,
                    )
                )
            from src.services.projects import lifecycle as lm, registry as rm
            from src.services.projects.clone import clone_settings_into

            saved = (lm._resolve_existing, lm._get_mutable_record, lm.write_record)
            saved_reg = rm.get_project_registry
            lm._resolve_existing = AsyncMock(return_value=source)
            lm._get_mutable_record = AsyncMock(return_value=(target, 1, 1))
            lm.write_record = AsyncMock(return_value=None)
            reg = MagicMock()
            reg.ensure_fresh = AsyncMock(return_value=None)
            rm.get_project_registry = lambda: reg
            outcome = 'ok'
            try:
                try:
                    asyncio.run(
                        clone_settings_into(
                            client,
                            slug=target.slug,
                            from_slug=source.slug,
                            axes=None,
                            expected_revision=1,
                        )
                    )
                except Exception as exc:
                    outcome = repr(exc)[:200]
            finally:
                lm._resolve_existing, lm._get_mutable_record, lm.write_record = saved
                rm.get_project_registry = saved_reg
            with bind_project(target):
                store = get_config_store()
                asyncio.run(store.refresh(client))
                pack, prof = live_pair()
            return outcome, pack, prof
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            curation_mod._default_curation_config = None
            reset_config_stores()


@pytest.mark.parametrize(
    'variant',
    ['source_profile_off', 'source_profile_registry_non_default'],
)
def test_clone_does_not_pair_stripped_pack_with_multi_region_default(
    ready_state, multi_default_registry, variant
):
    """Deployment default profile is multi-region. The source legitimately
    runs a STRIPPED pack because its profile is 'off' (or a registry
    single-region profile). _clone_activations skips that profile axis,
    so the target falls back to the multi-region default: the STRIPPED
    pack goes live under a multi-region profile."""
    profile_activation = (None, None) if variant == 'source_profile_off' else ('base', None)
    with contextlib.suppress(Exception):
        pass
    outcome, pack, prof = _clone_pair(profile_activation, plant_profile=True)
    print(
        variant,
        'clone ->',
        outcome,
        '| target live:',
        pack.class_system,
        prof and (prof.name, prof.max_regions_per_item),
    )
    assert not (
        pack.class_system == 'STRIPPED' and prof is not None and prof.max_regions_per_item > 1
    ), 'clone put a multi-box-stripped pack live under a multi-region profile in the target'


def test_active_source_stays_stored_for_deleted_but_activated_pack(
    app_client,
    ready_state,
    registry,
):
    """Activate pack_a, then delete its config doc directly (bypassing the
    router's own 'active pack can't be deleted' guard -- simulates the
    doc being removed out-of-band, e.g. an operator script). The
    activation's `revision` is still non-`None` (real, gate-validated
    when it was activated), so GET /active must still report
    ``source: 'stored'``, not misleadingly fall back to `'env'` just
    because the name no longer appears in the store's current pack
    list."""
    from src.config import get_curation_config
    from src.services.config_store.index import config_doc_id
    from src.services.config_store.store import reset_config_stores

    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': pack_body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code == 200

    idx = get_curation_config().configs_index
    doc_id = config_doc_id('prompt_pack', 'pack_a')  # current doc, no @rev
    asyncio.run(c.fake_os.delete(index=idx, id=doc_id))
    reset_config_stores()

    act = c.get(f'{PREFIX}/active').json()
    print('active after out-of-band delete:', act)
    assert act['active']['name'] == 'pack_a'
    assert act['source'] == 'stored', (
        'a deleted-but-still-activated pack must still report source=stored, not env'
    )
