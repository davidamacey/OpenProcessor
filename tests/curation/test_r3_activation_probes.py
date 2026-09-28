"""Round-3 reviewer probes, landed as permanent regression tests (round 4
confirmed all fixed -- see docs/design/openprocessor_internal/
w3_w4_review_2026-09-28.md). Each asserts CORRECT behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def app_client(monkeypatch):
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeConfigOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


PREFIX = '/curation/projects/default/prompt_packs'


def _body(cs=None):
    b = GENERIC_ITEM_PACK.to_dict()
    b.pop('name')
    if cs is not None:
        b['class_system'] = cs
    return b


def _activate_then_put(c):
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': _body()}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    r = c.put(f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': _body('R2 DRAFT')})
    assert r.status_code == 200, r.text


# ---- B1: VLM routes default path, served body AND stamp ----------------------
def test_r3_vlm_default_route_body_and_stamp(app_client):
    from src.routers.curation.vlm import _default_pack_name, _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _activate_then_put(app_client)
    name = asyncio.run(_default_pack_name(app_client.fake_os))
    lab = _get_vlm_labeler(name)
    print('route default', name, lab._pack.class_system[:20], prompt_pack_stamp(lab._pack))
    assert lab._pack.class_system == GENERIC_ITEM_PACK.class_system
    assert prompt_pack_stamp(lab._pack) == 'my_pack@1'


def test_r3_pipeline_default_body_and_stamp(app_client):
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, None))
    lab = _get_vlm_labeler(name, rev)
    print('pipeline default', name, rev, lab._pack.class_system[:20], prompt_pack_stamp(lab._pack))
    assert lab._pack.class_system == GENERIC_ITEM_PACK.class_system
    assert prompt_pack_stamp(lab._pack) == 'my_pack@1'


# ---- Provenance: per-run pin of the un-activated CURRENT revision -----------
def test_r3_per_run_pin_of_current_draft_is_stamped_with_its_own_revision(app_client):
    """§3.7: every VLM write stamps <name>@<revision> of the body that wrote it."""
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, 'my_pack@2'))
    lab = _get_vlm_labeler(name, rev)
    stamp = prompt_pack_stamp(lab._pack)
    print('per-run my_pack@2 served', lab._pack.class_system[:20], 'stamp', stamp)
    assert lab._pack.class_system == 'R2 DRAFT'
    assert stamp == 'my_pack@2', f'r2 body written under stamp {stamp}'


def test_r3_per_run_explicit_name_is_latest_per_spec(app_client):
    """§3.7: per-run `name` = latest saved revision."""
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, 'my_pack'))
    lab = _get_vlm_labeler(name, rev)
    print('per-run explicit my_pack ->', rev, lab._pack.class_system[:20])
    assert lab._pack.class_system == 'R2 DRAFT'


# ---- R1 -----------------------------------------------------------------------
def test_r3_r1_pinned_active_old_revision(app_client):
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, 'my_pack@1'))
    lab = _get_vlm_labeler(name, rev)
    assert (name, rev) == ('my_pack', 1)
    assert lab._pack.class_system == GENERIC_ITEM_PACK.class_system
    assert prompt_pack_stamp(lab._pack) == 'my_pack@1'


def test_r3_r1_far_past_revision(app_client):
    """r1 activated, r2 activated, r3 PUT. my_pack@1 exists in history."""
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': _body('R1')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': _body('R2')}).status_code
        == 200
    )
    r = c.post(
        f'{PREFIX}/my_pack/activate',
        json={'expected_active': {'name': 'my_pack', 'revision': 1}},
    )
    assert r.status_code == 200, r.text
    assert (
        c.put(f'{PREFIX}/my_pack', json={'expected_revision': 2, 'body': _body('R3')}).status_code
        == 200
    )
    # m5 fix (W3/W4 round-4 review): this test used to be "record-only"
    # (no assertion -- it just printed the outcome for a human to read).
    # By now `my_pack` is at CURRENT revision 3, and the currently
    # ACTIVE revision is 2 (rev 1 was superseded by activating rev 2
    # over it). Per `resolve_run_prompt_pack`'s own contract, this
    # process only resolves a per-run pin against its process-cached
    # current doc plus the store's currently-activated-revision pin --
    # NOT a full historical lookup. So rev 1 -- a real revision that
    # once existed and was even once active, but is neither current nor
    # currently-active anymore -- is exactly as unresolvable as the
    # never-saved rev 9. Both must 422 with the same "not resolvable in
    # this process" wording.
    for far_past in ('my_pack@1', 'my_pack@9'):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(resolve_run_prompt_pack(c.fake_os, far_past))
        assert exc_info.value.status_code == 422
        assert 'not resolvable in this process' in exc_info.value.detail['error']
    # record-only (no assert): scope question


# ---- Fail-closed: transient pinned-fetch error on a WARM store ---------------
def test_r3_warm_store_transient_error_keeps_last_good(app_client, monkeypatch):
    from src.services.config_store import get_config_store
    from src.services.labeling.vlm_prompts import active_prompt_pack, get_prompt_pack

    c = app_client
    _activate_then_put(c)
    store = get_config_store()
    asyncio.run(store.refresh(c.fake_os))
    assert active_prompt_pack().class_system == GENERIC_ITEM_PACK.class_system
    # another process bumps the revision (e.g. unrelated pack created)
    assert c.post(PREFIX, json={'name': 'other', 'body': _body('OTHER')}).status_code == 201
    fake = c.fake_os
    orig_get = fake.get

    async def flaky_get(*a, **kw):
        if str(kw.get('id', '')).endswith('@1'):
            raise ConnectionError('transient')
        return await orig_get(*a, **kw)

    monkeypatch.setattr(fake, 'get', flaky_get)
    # force a reload
    from dataclasses import replace

    store.current = replace(store.current, config_revision=-1)
    asyncio.run(store.refresh(fake))
    print('stale', store.current.stale, 'served', active_prompt_pack().class_system[:20])
    assert store.current.stale is True
    assert active_prompt_pack().class_system == GENERIC_ITEM_PACK.class_system
    warm_store_pack = get_prompt_pack('my_pack')
    assert warm_store_pack is not None
    assert warm_store_pack.class_system == GENERIC_ITEM_PACK.class_system


# ---- Genuine 404 on the pinned copy: still falls open? -----------------------
def test_r3_pinned_copy_404_does_not_serve_draft(app_client):
    from src.services.config_store import get_config_store
    from src.services.config_store.index import config_doc_id
    from src.services.labeling.vlm_prompts import active_prompt_pack, prompt_pack_stamp

    c = app_client
    _activate_then_put(c)
    fake = c.fake_os
    idx = get_config_store().index
    fake._docs[idx].pop(config_doc_id('prompt_pack', 'my_pack', 1))
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    asyncio.run(get_config_store().refresh(fake))
    served = active_prompt_pack()
    print('404 pinned copy -> served', served.class_system[:20], prompt_pack_stamp(served))
    assert served.class_system != 'R2 DRAFT'


# ---- activate route: transient error in the pinned fetch after the write -----
def test_r3_rollback_transient_error_status(app_client, monkeypatch):
    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': _body('A1')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'pack_b', 'body': _body('B1')}).status_code == 201
    assert c.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(f'{PREFIX}/pack_a', json={'expected_revision': 1, 'body': _body('A2')}).status_code
        == 200
    )
    assert (
        c.post(
            f'{PREFIX}/pack_b/activate', json={'expected_active': {'name': 'pack_a', 'revision': 1}}
        ).status_code
        == 200
    )
    fake = c.fake_os
    orig_get = fake.get
    # R4-1 fix means `pack_a@1`'s revision-copy doc is now fetched TWICE
    # on this path: once by the shared activation gate (to validate the
    # body before writing anything), and once more by
    # `_resolve_active_body` inside `activate_and_apply` (to pin the
    # exact body onto the write). Fail only the SECOND fetch, so the gate
    # itself succeeds and the transient error is isolated to exactly the
    # N4 code path this test targets (inside `activate_and_apply`, after
    # the gate has already cleared).
    calls = {'n': 0}

    async def flaky_get(*a, **kw):
        if str(kw.get('id', '')).endswith('pack_a@1'):
            calls['n'] += 1
            if calls['n'] >= 2:
                raise ConnectionError('transient')
        return await orig_get(*a, **kw)

    monkeypatch.setattr(fake, 'get', flaky_get)
    tc = TestClient(c.app, raise_server_exceptions=False)
    r = tc.post(
        f'{PREFIX}/active/rollback', json={'expected_active': {'name': 'pack_b', 'revision': 1}}
    )
    monkeypatch.setattr(fake, 'get', orig_get)
    act = c.get(f'{PREFIX}/active').json()
    # m3 fix (W3/W4 round-4 review): the OLD assertion here read
    # `GET /active`, which is served from the in-process `ConfigStore`
    # snapshot -- stale in BOTH the pre-fix and post-fix worlds (this
    # process never applied the failed write locally either way), so it
    # stayed green even with N4's fix reverted (write-then-resolve
    # restored). The only thing that actually distinguishes "aborted
    # before commit" from "committed, then 500'd" is the STORED
    # `activation:*` doc itself.
    import asyncio

    from src.config import get_curation_config
    from src.services.config_store.index import get_activation

    stored = asyncio.run(get_activation(fake, get_curation_config().configs_index, 'prompt_pack'))
    print(
        'rollback under transient ->',
        r.status_code,
        r.text[:160],
        '| persisted active (stale snapshot)',
        act,
        '| stored activation doc',
        stored,
    )
    # N4 fix: resolving the pinned body before writing the activation
    # means a transient error aborts BEFORE anything commits -- the
    # STORED doc must still name pack_b (unchanged), not silently hold a
    # half-committed pack_a activation the caller was told (via a 500)
    # never happened.
    assert stored is not None
    assert stored.get('name') == 'pack_b'


# ---- PUT /settings bridge: activation gate bypass? ---------------------------
def test_r3_settings_bridge_runs_activation_gate(app_client, monkeypatch):
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    stripped = _body()
    stripped['combined_system'] = (
        'Return JSON keys class_id class_confidence region_visible region_bbox_correct region_confidence'
    )
    stripped['combined_batch_system'] = (
        'Return JSON results with img class_id class_confidence region_visible region_bbox_correct region_confidence'
    )
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(f'{PREFIX}/good', json={'expected_revision': 1, 'body': stripped}).status_code == 200
    )
    r = c.post(
        f'{PREFIX}/good/activate',
        json={'expected_active': {'name': 'good', 'revision': 1}, 'force': True},
    )
    assert r.status_code == 422  # the gate on the activate route
    r = c.put('/curation/projects/default/settings', json={'defaults': {'prompt_pack': 'good'}})
    act = c.get(f'{PREFIX}/active').json()['active']
    served = active_prompt_pack()
    print(
        'PUT /settings ->',
        r.status_code,
        r.text[:160],
        '| active',
        act,
        '| served multi-box-stripped?',
        'region_bbox_correct region_confidence' in served.combined_system,
    )
    assert not (r.status_code == 200 and act.get('revision') == 2), 'settings bridge bypassed gate'
