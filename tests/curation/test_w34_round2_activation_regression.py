"""W3/W4 review round-2 (2026-09-28) regression tests: the API's VLM write
routes and the pipeline's omitted-``prompt_pack`` default must both resolve
through the config store's activation-pinned body (B1, closing the
remaining bypass after round 1 only fixed
``active_prompt_pack()``/``get_active_region_profile()`` callers), a
transient pinned-revision fetch failure must fail closed rather than
serving an un-activated PUT body, and ``name@<past-activated-revision>``
must resolve and label successfully at request time (the R1 regression
this round's own B1 fix introduced). Landed here as permanent tests
(moved from the reviewer's throwaway probe file) rather than left as
scratch fixtures."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
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
    client.fake_os = fake_os
    return client


PREFIX = '/curation/projects/default/prompt_packs'


def _body():
    b = GENERIC_ITEM_PACK.to_dict()
    b.pop('name')
    return b


def _activate_then_put(app_client):
    assert app_client.post(PREFIX, json={'name': 'my_pack', 'body': _body()}).status_code == 201
    assert (
        app_client.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code
        == 200
    )
    b2 = _body()
    b2['class_system'] = 'EDITED AFTER ACTIVATION'
    r = app_client.put(f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': b2})
    assert r.status_code == 200, r.text


def test_c1_api_vlm_default_labeler_serves_pinned_body(app_client):
    """/vlm/label_batch & co use _get_vlm_labeler(await _default_pack_name(os))."""
    from src.routers.curation.vlm import _default_pack_name, _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _activate_then_put(app_client)
    name = asyncio.run(_default_pack_name(app_client.fake_os))
    labeler = _get_vlm_labeler(name)
    stamp = prompt_pack_stamp(labeler._pack)
    print('default name', name, 'served', labeler._pack.class_system[:30], 'stamp', stamp)
    assert labeler._pack.class_system == GENERIC_ITEM_PACK.class_system, (
        f'API VLM route default path served the un-activated r2 body, stamped {stamp}'
    )


def test_c1b_pipeline_omitted_prompt_pack_serves_pinned_body(app_client):
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, None))
    labeler = _get_vlm_labeler(name, rev)
    print('pipeline default', name, rev, labeler._pack.class_system[:30])
    assert labeler._pack.class_system == GENERIC_ITEM_PACK.class_system


def test_c1c_resolve_active_body_fetch_failure_does_not_serve_put_body(app_client, monkeypatch):
    """Cold reload of the snapshot when the pinned @rev GET fails transiently."""
    from src.services.config_store import get_config_store
    from src.services.config_store.store import reset_config_stores
    from src.services.labeling.vlm_prompts import active_prompt_pack

    _activate_then_put(app_client)
    reset_config_stores()
    fake = app_client.fake_os
    orig_get = fake.get

    async def flaky_get(*a, **kw):
        if '@1' in str(kw.get('id', '')):
            raise ConnectionError('transient')
        return await orig_get(*a, **kw)

    monkeypatch.setattr(fake, 'get', flaky_get)
    asyncio.run(get_config_store().refresh(fake))
    served = active_prompt_pack()
    print('after flaky reload served', served.class_system[:30], get_config_store().current.stale)
    assert served.class_system != 'EDITED AFTER ACTIVATION'


def test_c2_put_on_active_pack_marks_new_revision_and_active_rev_unchanged(app_client):
    _activate_then_put(app_client)
    active = app_client.get(f'{PREFIX}/active').json()
    doc = app_client.get(f'{PREFIX}/my_pack').json()
    print(active, {k: doc.get(k) for k in ('revision', 'active', 'active_revision')})
    assert active['active']['revision'] == 1
    assert doc['revision'] == 2


def test_c3_reactivate_latest_revalidates(app_client, monkeypatch):
    """After PUT, activating the new revision must go through for_activation gate."""
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile

    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    stripped = _body()
    stripped['combined_system'] = (
        'Return JSON keys class_id class_confidence region_visible region_bbox_correct region_confidence'
    )
    stripped['combined_batch_system'] = (
        'Return JSON results with img class_id class_confidence region_visible region_bbox_correct region_confidence'
    )
    assert app_client.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
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


def test_c4_pinned_body_survives_rollback(app_client):
    from src.services.labeling.vlm_prompts import active_prompt_pack

    assert app_client.post(PREFIX, json={'name': 'pack_a', 'body': _body()}).status_code == 201
    assert app_client.post(PREFIX, json={'name': 'pack_b', 'body': _body()}).status_code == 201
    assert (
        app_client.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code
        == 200
    )
    b2 = _body()
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


def test_c5_template_profile_and_pack_activation(app_client):
    r = app_client.post(f'{PREFIX}/vehicle_wheel/activate', json={'expected_active': None})
    print(r.status_code, r.text[:200])
    assert r.status_code == 403
    assert r.json()['detail']['error'] == 'read_only' if 'detail' in r.json() else True


def test_c6_put_and_clone_name_rules(app_client):
    for bad in ('off', 'default', 'none', 'Not_A_Slug', 'Bad Name'):
        r = app_client.put(f'{PREFIX}/{bad}', json={'expected_revision': 0, 'body': _body()})
        assert r.status_code in (404, 422), (bad, r.status_code)
    assert app_client.post(PREFIX, json={'name': 'srcp', 'body': _body()}).status_code == 201
    for bad in ('off', 'default', 'none', 'Not_A_Slug', 'Bad Name'):
        r = app_client.post(f'{PREFIX}/srcp/clone', json={'new_name': bad})
        print('clone', bad, r.status_code)
        assert r.status_code == 422, (bad, r.status_code, r.text[:200])
    r = app_client.post(f'{PREFIX}/srcp/clone', json={'new_name': 'srcp'})
    assert r.status_code == 409


def test_c7_profile_clone_and_put_reserved_names(app_client, monkeypatch):
    from curation.test_w34_regression_probes import _profile_body

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://nowhere:1')
    base = '/curation/projects/default/region_profiles'
    assert (
        app_client.post(base, json={'name': 'srcprof', 'body': _profile_body()}).status_code == 201
    )
    for bad in ('off', 'default', 'none', 'Not_A_Slug'):
        r = app_client.put(f'{base}/{bad}', json={'expected_revision': 0, 'body': _profile_body()})
        assert r.status_code in (404, 422), (bad, r.status_code)
        r = app_client.post(f'{base}/srcprof/clone', json={'new_name': bad})
        print('profile clone', bad, r.status_code)
        assert r.status_code == 422, (bad, r.status_code, r.text[:200])


def test_c8_put_on_active_profile_does_not_change_runtime(app_client, monkeypatch):
    from curation.test_w34_regression_probes import _profile_body
    from src.services.detection.profile_registry import get_active_region_profile

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://nowhere:1')
    base = '/curation/projects/default/region_profiles'
    assert (
        app_client.post(base, json={'name': 'liveprof', 'body': _profile_body()}).status_code == 201
    )
    r = app_client.post(f'{base}/liveprof/activate', json={'expected_active': None, 'force': True})
    assert r.status_code == 200, r.text
    b2 = _profile_body()
    b2['max_regions_per_item'] = 7
    b2['display_name'] = 'EDITED'
    r = app_client.put(f'{base}/liveprof', json={'expected_revision': 1, 'body': b2})
    assert r.status_code == 200, r.text
    p = get_active_region_profile()
    assert p is not None
    print('active profile', p.name, p.max_regions_per_item, p.display_name)
    assert p.max_regions_per_item != 7
    assert p.display_name != 'EDITED'


def test_c9_per_run_name_at_past_revision(app_client):
    """§3.7: ?prompt_pack=name@<rev> pins that exact revision."""
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, 'my_pack@1'))
    print('resolved', name, rev)
    try:
        labeler = _get_vlm_labeler(name, rev)
    except ValueError as exc:
        pytest.fail(
            f'valid past revision my_pack@1 accepted at request time but labeler raises: {exc}'
        )
    assert labeler._pack.class_system == GENERIC_ITEM_PACK.class_system
