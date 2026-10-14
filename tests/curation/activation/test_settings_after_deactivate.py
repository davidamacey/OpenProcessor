"""PUT /settings can switch an axis back on after it was turned off, by either route,
including from a process whose store is cold."""

from __future__ import annotations

import pytest

from curation.activation.constants import PREFIX, RP, SETTINGS, pack_body


pytestmark = pytest.mark.usefixtures('vlm_env')


@pytest.mark.parametrize('axis', ['prompt_pack', 'detection_profile'])
def test_settings_reactivate_after_off(app_client, monkeypatch, axis):
    from src.config import DetectionProfile
    from src.services.detection import profile_registry
    from src.services.triton_control import TritonControlService

    async def _repo():
        return [{'name': 'wheel_detector', 'state': 'READY', 'version': '1'}]

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())
    profile_registry._reset_registry_for_tests()
    profile_registry.register_profile(
        DetectionProfile(name='wheel', detector_model='wheel_detector', text_reader='none'),
        default=True,
    )
    c = app_client
    try:
        assert c.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
        val = 'good' if axis == 'prompt_pack' else 'wheel'
        r1 = c.put(SETTINGS, json={'defaults': {axis: val}})
        r2 = c.put(SETTINGS, json={'defaults': {axis: None}})
        r3 = c.put(SETTINGS, json={'defaults': {axis: val}})
        print(axis, r1.status_code, r2.status_code, r3.status_code, r3.text[:120])
        assert (r1.status_code, r2.status_code, r3.status_code) == (200, 200, 200)
    finally:
        profile_registry._reset_registry_for_tests()


def test_activate_route_after_deactivate_with_served_active(app_client):
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': None}}).status_code == 200
    served = c.get(f'{PREFIX}/active').json()
    print('GET /active after off:', {k: served.get(k) for k in ('active', 'etag', 'previous')})
    r = c.post(f'{PREFIX}/good/activate', json={'expected_active': served.get('active')})
    print('activate with served active ->', r.status_code, r.text[:150])
    assert r.status_code == 200


@pytest.mark.parametrize('axis', ['prompt_pack', 'detection_profile'])
def test_settings_switch_target_through_none(app_client, ready_state, registry, axis):
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
    assert c.post(PREFIX, json={'name': 'good2', 'body': pack_body('G2')}).status_code == 201
    x, y = ('good', 'good2') if axis == 'prompt_pack' else ('base', 'other')
    rs = [
        c.put(SETTINGS, json={'defaults': {axis: x}}),
        c.put(SETTINGS, json={'defaults': {axis: None}}),
        c.put(SETTINGS, json={'defaults': {axis: y}}),
        c.put(SETTINGS, json={'defaults': {axis: 'off' if axis == 'detection_profile' else None}}),
        c.put(SETTINGS, json={'defaults': {axis: x}}),
    ]
    print(axis, [r.status_code for r in rs], rs[2].text[:150])
    assert [r.status_code for r in rs] == [200] * 5


def test_settings_after_deactivate_route(app_client, ready_state, registry):
    c = app_client
    assert c.post(f'{RP}/base/activate', json={'expected_active': None}).status_code == 200
    d = c.post(f'{RP}/deactivate', json={'expected_active': {'name': 'base', 'revision': None}})
    assert d.status_code == 200, d.text
    r = c.put(SETTINGS, json={'defaults': {'detection_profile': 'other'}})
    print('after /deactivate ->', r.status_code, r.text[:160])
    assert r.status_code == 200


def test_settings_on_cold_store_after_off(app_client, ready_state, registry):
    """Another process deactivated; this process's store is reset (cold)."""
    from src.services.config_store.store import reset_config_stores

    c = app_client
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': None}}).status_code == 200
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': None}}).status_code == 200
    reset_config_stores()
    r = c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}})
    print('cold after off ->', r.status_code, r.text[:160])
    assert r.status_code == 200
