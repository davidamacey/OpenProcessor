"""A pack and a detection profile are gated as a pair: the multi-box-stripped pack never
goes live under a multi-region profile, whichever route or order writes them."""

from __future__ import annotations

import pytest

from curation.activation.constants import PREFIX, SETTINGS, pack_body, stripped_pack_body
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


pytestmark = pytest.mark.usefixtures('vlm_env')


def test_settings_two_axes_failure_does_not_commit_first_axis(app_client, monkeypatch):
    from src.config import DetectionProfile
    from src.services.detection import profile_registry
    from src.services.triton_control import TritonControlService

    async def _repo():
        return []  # detector not READY -> profile gate fails

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())
    profile_registry._reset_registry_for_tests()
    profile_registry.register_profile(
        DetectionProfile(name='wheel', detector_model='wheel_detector', text_reader='none'),
        default=True,
    )
    c = app_client
    try:
        assert c.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
        r = c.put(
            SETTINGS, json={'defaults': {'prompt_pack': 'good', 'detection_profile': 'wheel'}}
        )
        act = c.get(f'{PREFIX}/active').json()['active']
        print('two-axis PUT ->', r.status_code, '| pack active after 422:', act)
        assert not (r.status_code == 422 and act and act.get('name') == 'good'), (
            'PUT /settings 422d but committed the prompt_pack activation'
        )
    finally:
        profile_registry._reset_registry_for_tests()


@pytest.mark.parametrize('order', ['pack_first', 'profile_first'])
def test_two_axis_put_cannot_pair_stripped_pack_with_multi_profile(
    app_client, ready_state, registry, order
):
    from src.services.detection.profile_registry import get_active_region_profile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body('GOOD')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'single', 'body': stripped_pack_body()}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200

    # Controls: each half of the pair is refused on its own once the other is live.
    defaults = (
        {'prompt_pack': 'single', 'detection_profile': 'multi'}
        if order == 'pack_first'
        else {'detection_profile': 'multi', 'prompt_pack': 'single'}
    )
    r = c.put(SETTINGS, json={'defaults': defaults})
    served_pack = active_prompt_pack()
    served_prof = get_active_region_profile()
    assert served_prof is not None
    print(
        order,
        '->',
        r.status_code,
        r.text[:200],
        '| live:',
        served_pack.class_system,
        served_prof.name,
        served_prof.max_regions_per_item,
    )
    assert not (served_pack.class_system == 'STRIPPED' and served_prof.max_regions_per_item > 1), (
        'two-axis PUT /settings put a multi-box-stripped pack live under a multi-region profile'
    )


def test_single_axis_activate_refuses_stripped_pack_under_multi_profile(
    app_client, ready_state, registry
):
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body('GOOD')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'single', 'body': stripped_pack_body()}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}}).status_code == 200
    r = c.post(
        f'{PREFIX}/single/activate',
        json={'expected_active': {'name': 'good', 'revision': 1}, 'force': True},
    )
    assert r.status_code == 422


def test_reactivating_edited_revision_revalidates_gate(app_client, monkeypatch):
    """After PUT, activating the new revision must go through for_activation gate."""
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile

    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    stripped = pack_body()
    stripped['combined_system'] = (
        'Return JSON keys class_id class_confidence region_visible region_bbox_correct region_confidence'
    )
    stripped['combined_batch_system'] = (
        'Return JSON results with img class_id class_confidence region_visible region_bbox_correct region_confidence'
    )
    assert app_client.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
    assert (
        app_client.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code
        == 200
    )
    assert (
        app_client.put(
            f'{PREFIX}/good', json={'expected_revision': 1, 'body': stripped}
        ).status_code
        == 200
    )
    r = app_client.post(
        f'{PREFIX}/good/activate',
        json={'expected_active': {'name': 'good', 'revision': 1}, 'force': True},
    )
    print('reactivate stripped r2 ->', r.status_code)
    assert r.status_code == 422


def test_put_on_active_pack_does_not_change_running_pack(app_client, monkeypatch) -> None:
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile

    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    stripped = pack_body()
    stripped['combined_system'] = (
        'Return JSON keys class_id class_confidence region_visible region_bbox_correct '
        'region_confidence'
    )
    stripped['combined_batch_system'] = (
        'Return JSON results with img class_id class_confidence region_visible '
        'region_bbox_correct region_confidence'
    )
    assert app_client.post(PREFIX, json={'name': 'bad', 'body': stripped}).status_code == 201
    r_bad = app_client.post(f'{PREFIX}/bad/activate', json={'expected_active': None, 'force': True})
    print('activate stripped pack directly ->', r_bad.status_code)
    assert app_client.post(PREFIX, json={'name': 'good', 'body': pack_body()}).status_code == 201
    r = app_client.post(f'{PREFIX}/good/activate', json={'expected_active': None})
    print('activate good ->', r.status_code)
    r_put = app_client.put(f'{PREFIX}/good', json={'expected_revision': 1, 'body': stripped})
    from src.services.labeling.vlm_prompts import active_prompt_pack

    served = active_prompt_pack()
    print(
        'PUT stripped onto ACTIVE pack ->',
        r_put.status_code,
        '| served combined_system:',
        served.combined_system[:40],
    )
    assert r_bad.status_code == 422
    # any_domain_plan.md §4.4: "PUT ... on the active profile does NOT
    # change the active revision" -- the write itself is allowed (it
    # writes a new revision the operator can review/activate later); the
    # thing that must never happen is the stripped body going live
    # without a separate activate call re-running the for-activation gate.
    assert r_put.status_code == 200, r_put.text
    assert served.combined_system == GENERIC_ITEM_PACK.combined_system, (
        'PUT on the active pack changed what is actually running'
    )
