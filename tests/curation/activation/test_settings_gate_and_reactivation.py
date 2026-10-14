"""PUT /settings runs the same activation gate as the activate routes."""

from __future__ import annotations

import asyncio

import pytest

from curation.activation.constants import PREFIX, SETTINGS, pack_body, stripped_pack_body


pytestmark = pytest.mark.usefixtures('vlm_env')


def test_settings_pack_gate_matches_activate_route(app_client, monkeypatch):
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile

    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/good', json={'expected_revision': 1, 'body': stripped_pack_body()}
        ).status_code
        == 200
    )
    r1 = c.post(
        f'{PREFIX}/good/activate',
        json={'expected_active': {'name': 'good', 'revision': 1}, 'force': True},
    )
    r2 = c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}})
    print(
        'activate',
        r1.status_code,
        r1.json().get('detail', {}).get('error') if r1.status_code != 200 else '',
    )
    print('settings', r2.status_code, r2.text[:300])
    assert r1.status_code == 422
    assert r2.status_code == 422
    codes_a = sorted(e['code'] for e in r1.json()['detail']['report']['errors'])
    codes_b = sorted(e['code'] for e in r2.json()['detail']['report']['errors'])
    print(codes_a, codes_b)
    assert codes_a == codes_b
    assert c.get(f'{PREFIX}/active').json()['active']['revision'] == 1


@pytest.mark.parametrize('ready', [True, False])
def test_settings_profile_gate_matches_activate_route(app_client, monkeypatch, ready):
    from src.config import DetectionProfile
    from src.services.detection import profile_registry
    from src.services.triton_control import TritonControlService

    async def _repo():
        return [{'name': 'wheel_detector', 'state': 'READY', 'version': '1'}] if ready else []

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())
    profile_registry._reset_registry_for_tests()
    profile_registry.register_profile(
        DetectionProfile(name='wheel', detector_model='wheel_detector', text_reader='none'),
        default=True,
    )
    try:
        r_act = app_client.post(
            '/curation/projects/default/region_profiles/wheel/activate',
            json={'expected_active': None, 'force': True},
        )
        # reset activation so both paths start from the same state
        if r_act.status_code == 200:
            app_client.post(
                '/curation/projects/default/region_profiles/deactivate',
                json={'expected_active': {'name': 'wheel', 'revision': None}},
            )
        r_set = app_client.put(SETTINGS, json={'defaults': {'detection_profile': 'wheel'}})
        print(
            'ready',
            ready,
            'activate',
            r_act.status_code,
            'settings',
            r_set.status_code,
            r_set.text[:200],
        )
        if ready:
            assert r_set.status_code == 200, r_set.text
        else:
            assert r_act.status_code == 422
            assert r_set.status_code == 422
            assert r_set.json()['detail']['error'] == 'validation_failed'
    finally:
        profile_registry._reset_registry_for_tests()


def test_settings_bridge_runs_activation_gate(app_client, monkeypatch):
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    stripped = pack_body()
    stripped['combined_system'] = (
        'Return JSON keys class_id class_confidence region_visible region_bbox_correct region_confidence'
    )
    stripped['combined_batch_system'] = (
        'Return JSON results with img class_id class_confidence region_visible region_bbox_correct region_confidence'
    )
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
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


def test_legacy_settings_doc_prompt_pack_does_not_override_activation(app_client):
    """A settings doc still carrying defaults.prompt_pack (no startup
    migration strips it) vs. a pack activated through the store."""
    from src.clients.curation_opensearch.settings_doc import update_curation_settings
    from src.routers.curation.vlm import _default_pack_name
    from src.services.labeling.run_labeler import get_vlm_labeler

    c = app_client
    assert c.post(PREFIX, json={'name': 'legacy', 'body': pack_body('LEGACY')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body('GOOD')}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    asyncio.run(update_curation_settings(c.fake_os, {'prompt_pack': 'legacy'}))
    shown = c.get(SETTINGS).json()['defaults'].get('prompt_pack')
    name = asyncio.run(_default_pack_name(c.fake_os))
    served = get_vlm_labeler(name)._pack.class_system
    print('GET /settings shows', shown, '| VLM routes default name', name, '| serves', served)
    assert served == 'GOOD'
