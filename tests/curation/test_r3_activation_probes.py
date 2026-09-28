"""Round-3 reviewer probes (scratch copy only). Each asserts CORRECT behavior."""

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
    for req in ('my_pack@1', 'my_pack@9'):
        try:
            out = asyncio.run(resolve_run_prompt_pack(c.fake_os, req))
            print(req, '->', out)
        except HTTPException as exc:
            print(req, '->', exc.status_code, exc.detail)
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

    async def flaky_get(*a, **kw):
        if str(kw.get('id', '')).endswith('pack_a@1'):
            raise ConnectionError('transient')
        return await orig_get(*a, **kw)

    monkeypatch.setattr(fake, 'get', flaky_get)
    tc = TestClient(c.app, raise_server_exceptions=False)
    r = tc.post(
        f'{PREFIX}/active/rollback', json={'expected_active': {'name': 'pack_b', 'revision': 1}}
    )
    monkeypatch.setattr(fake, 'get', orig_get)
    act = c.get(f'{PREFIX}/active').json()
    print('rollback under transient ->', r.status_code, r.text[:160], '| persisted active', act)
    # N4 fix: resolving the pinned body before writing the activation
    # means a transient error aborts BEFORE anything commits -- the
    # rollback must still be reporting pack_b (unchanged), not silently
    # holding a half-committed pack_a activation the caller was told
    # (via a 500) never happened.
    assert act['active']['name'] == 'pack_b'


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
