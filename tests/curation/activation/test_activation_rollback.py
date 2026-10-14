"""Rollback re-runs the activation gate, restores the pinned body of the previous
revision, and aborts before committing when the pinned body cannot be fetched."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from curation.activation.constants import PREFIX, RP, pack_body, stripped_pack_body
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


pytestmark = pytest.mark.usefixtures('vlm_env')


def test_rollback_transient_error_aborts_before_commit(app_client, monkeypatch):
    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': pack_body('A1')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'pack_b', 'body': pack_body('B1')}).status_code == 201
    assert c.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/pack_a', json={'expected_revision': 1, 'body': pack_body('A2')}
        ).status_code
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
    # `pack_a@1`'s revision-copy doc is fetched TWICE on this path: once by the shared
    # activation gate (to validate the body before writing anything), and once more by
    # `_resolve_active_body` inside `activate_and_apply` (to pin the exact body onto the
    # write). Fail only the SECOND fetch, so the gate itself succeeds and the transient
    # error is isolated to the code inside `activate_and_apply`, after the gate cleared.
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
    # `GET /active` is served from the in-process `ConfigStore` snapshot, which this
    # process never updates for the failed write either way, so it cannot tell "aborted
    # before commit" from "committed, then 500'd". Only the STORED `activation:*` doc can.
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
    # Resolving the pinned body before writing the activation means a transient error
    # aborts BEFORE anything commits: the STORED doc must still name pack_b, not hold a
    # half-committed pack_a activation the caller was told (via a 500) never happened.
    assert stored is not None
    assert stored.get('name') == 'pack_b'


def test_pack_rollback_cannot_bypass_multibox_gate(app_client, monkeypatch):
    """A single-box pack S was validly active under a single-region profile.
    Then pack G (multi-box keys) was activated, then the profile went
    multi-region (allowed: G has the keys). Rolling the pack back to S puts a
    pack live that `POST /S/activate` refuses even with force."""
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'single', 'body': stripped_pack_body()}).status_code == 201
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
    r = c.post(f'{PREFIX}/single/activate', json={'expected_active': None})
    assert r.status_code == 200, r.text  # no multi-region profile active yet
    r = c.post(
        f'{PREFIX}/good/activate', json={'expected_active': {'name': 'single', 'revision': 1}}
    )
    assert r.status_code == 200, r.text
    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    direct = c.post(
        f'{PREFIX}/single/activate',
        json={'expected_active': {'name': 'good', 'revision': 1}, 'force': True},
    )
    rb = c.post(
        f'{PREFIX}/active/rollback', json={'expected_active': {'name': 'good', 'revision': 1}}
    )
    served = active_prompt_pack()
    print(
        'direct activate single ->',
        direct.status_code,
        '| rollback ->',
        rb.status_code,
        '| served',
        served.class_system,
    )
    assert direct.status_code == 422
    assert not (rb.status_code == 200 and served.class_system == 'STRIPPED'), (
        'rollback activated a pack the never-bypassable gate refuses'
    )


def test_profile_rollback_refuses_profile_whose_detector_is_gone(app_client, ready_state, registry):
    from src.services.detection.profile_registry import get_active_region_profile

    c = app_client
    r = c.post(f'{RP}/base/activate', json={'expected_active': None})
    assert r.status_code == 200, r.text
    r = c.post(f'{RP}/other/activate', json={'expected_active': {'name': 'base', 'revision': None}})
    assert r.status_code == 200, r.text
    ready_state['ready'] = {'other_detector'}  # wheel_detector unloaded
    direct = c.post(
        f'{RP}/base/activate',
        json={'expected_active': {'name': 'other', 'revision': None}, 'force': True},
    )
    rb = c.post(
        f'{RP}/active/rollback', json={'expected_active': {'name': 'other', 'revision': None}}
    )
    print('direct', direct.status_code, '| rollback', rb.status_code, rb.text[:200])
    assert direct.status_code == 422
    assert rb.status_code == 422, 'profile rollback bypassed the detector-READY gate'
    current_profile = get_active_region_profile()
    assert current_profile is not None
    assert current_profile.name == 'other'


def test_rollback_to_superseded_revision_serves_pinned_body(app_client):
    from src.services.labeling.vlm_prompts import active_prompt_pack, prompt_pack_stamp

    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': pack_body('A1')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'pack_b', 'body': pack_body('B1')}).status_code == 201
    assert c.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/pack_a', json={'expected_revision': 1, 'body': pack_body('A2 DRAFT')}
        ).status_code
        == 200
    )
    assert (
        c.post(
            f'{PREFIX}/pack_b/activate', json={'expected_active': {'name': 'pack_a', 'revision': 1}}
        ).status_code
        == 200
    )
    rb = c.post(
        f'{PREFIX}/active/rollback', json={'expected_active': {'name': 'pack_b', 'revision': 1}}
    )
    served = active_prompt_pack()
    print(
        'rollback ->',
        rb.status_code,
        rb.text[:200],
        '| served',
        served.class_system,
        prompt_pack_stamp(served),
    )
    assert rb.status_code == 200
    assert served.class_system == 'A1'
    assert prompt_pack_stamp(served) == 'pack_a@1'


def test_rollback_to_superseded_revision_that_fails_gate_is_refused(app_client, monkeypatch):
    """The fallback must not let a superseded revision skip validation:
    pack_a@1 is STRIPPED, pack_a current r2 is GOOD. After the profile goes
    multi-region, rolling back to pack_a@1 must 422 even though the CURRENT
    pack_a (r2) would pass."""
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': stripped_pack_body()}).status_code == 201
    assert c.post(PREFIX, json={'name': 'pack_b', 'body': pack_body('B1')}).status_code == 201
    assert c.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/pack_a', json={'expected_revision': 1, 'body': pack_body('A2 GOOD')}
        ).status_code
        == 200
    )
    assert (
        c.post(
            f'{PREFIX}/pack_b/activate', json={'expected_active': {'name': 'pack_a', 'revision': 1}}
        ).status_code
        == 200
    )
    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    rb = c.post(
        f'{PREFIX}/active/rollback', json={'expected_active': {'name': 'pack_b', 'revision': 1}}
    )
    print(
        'rollback ->', rb.status_code, rb.text[:200], '| served', active_prompt_pack().class_system
    )
    assert rb.status_code == 422
    assert active_prompt_pack().class_system == 'B1'


def test_pinned_body_survives_rollback(app_client):
    from src.services.labeling.vlm_prompts import active_prompt_pack

    assert app_client.post(PREFIX, json={'name': 'pack_a', 'body': pack_body()}).status_code == 201
    assert app_client.post(PREFIX, json={'name': 'pack_b', 'body': pack_body()}).status_code == 201
    assert (
        app_client.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code
        == 200
    )
    b2 = pack_body()
    b2['class_system'] = 'A R2'
    assert (
        app_client.put(f'{PREFIX}/pack_a', json={'expected_revision': 1, 'body': b2}).status_code
        == 200
    )
    assert (
        app_client.post(
            f'{PREFIX}/pack_b/activate', json={'expected_active': {'name': 'pack_a', 'revision': 1}}
        ).status_code
        == 200
    )
    r = app_client.post(
        f'{PREFIX}/active/rollback', json={'expected_active': {'name': 'pack_b', 'revision': 1}}
    )
    print('rollback', r.status_code, r.text[:200])
    assert r.status_code == 200
    served = active_prompt_pack()
    print('after rollback served', served.name, served.class_system[:20])
    assert served.name == 'pack_a'
    assert served.class_system == GENERIC_ITEM_PACK.class_system
