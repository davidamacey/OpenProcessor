"""Round-4 reviewer probes (scratch copy only). Each asserts CORRECT behavior;
a red test is a real defect."""

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
    from src.clients import curation_opensearch
    from src.services.config_store.store import reset_config_stores

    curation_opensearch._settings_cache.clear()
    reset_config_stores()
    yield
    curation_opensearch._settings_cache.clear()
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
SETTINGS = '/curation/projects/default/settings'


def _body(cs=None):
    b = GENERIC_ITEM_PACK.to_dict()
    b.pop('name')
    if cs is not None:
        b['class_system'] = cs
    return b


def _stripped():
    b = _body('STRIPPED')
    b['combined_system'] = (
        'Return JSON keys class_id class_confidence region_visible region_bbox_correct '
        'region_confidence'
    )
    b['combined_batch_system'] = (
        'Return JSON results with img class_id class_confidence region_visible '
        'region_bbox_correct region_confidence'
    )
    return b


# ---- N1 baseline: prompt_pack via settings bridge refuses stripped ----------
def test_r4_n1_pack_settings_gate(app_client, monkeypatch):
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile

    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(f'{PREFIX}/good', json={'expected_revision': 1, 'body': _stripped()}).status_code
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


# ---- N1 profile axis via settings bridge ------------------------------------
@pytest.mark.parametrize('ready', [True, False])
def test_r4_n1_profile_settings_gate(app_client, monkeypatch, ready):
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


# ---- Sibling caller: rollback re-activates without the gate -----------------
def test_r4_rollback_bypasses_multibox_gate(app_client, monkeypatch):
    """A single-box pack S was validly active under a single-region profile.
    Then pack G (multi-box keys) was activated, then the profile went
    multi-region (allowed: G has the keys). Rolling the pack back to S puts a
    pack live that `POST /S/activate` refuses even with force."""
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'single', 'body': _stripped()}).status_code == 201
    assert c.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
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


# ---- Combined PUT /settings: second axis 422 after first axis committed -----
def test_r4_settings_two_axes_partial_write(app_client, monkeypatch):
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
        assert c.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
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


# ---- N7 + job re-resolution: a request-time pin is lost at job start --------
def test_r4_start_job_args_accepted_by_pipeline_fn():
    """The worker calls pipeline_fn(opensearch=, progress=, **trigger_args);
    /pipeline/auto_label/start puts 'prompt_pack_revision' in those args.

    R6-m1 fix (W3/W4 round-6 review): the worker's target is
    ``_run_auto_label`` now, not the public ``pipeline_auto_label`` route
    wrapper -- the route no longer accepts these params at all (they
    would otherwise be reachable from an HTTP request)."""
    import inspect

    from src.routers.curation.pipeline import _run_auto_label

    assert 'prompt_pack_revision' in inspect.signature(_run_auto_label).parameters
    assert 'prompt_pack_omitted' in inspect.signature(_run_auto_label).parameters


def test_r4_job_reresolution_keeps_request_pin(app_client, monkeypatch):
    """R4-3, rewritten from the original probe: the original assertion
    (calling ``resolve_run_prompt_pack`` a SECOND time on the bare name
    ``at_request[0]`` and expecting the same result) can never hold --
    N7 (confirmed correct by the round-4 reviewer) makes a bare per-run
    name resolve to the LATEST saved revision by design, so re-running
    the resolver on a bare name is *supposed* to move forward. The real
    bug was that ``pipeline_auto_label`` (the job body) called the
    resolver a second time at all once ``/start`` had already resolved
    and pinned ``(name, revision)``. This tests the actual fix: with
    ``prompt_pack_revision`` supplied (exactly as ``/start`` -> the
    worker -> ``pipeline_fn(**trigger_args)`` supplies it),
    ``pipeline_auto_label`` must use the pin as-is and must NOT call
    ``resolve_run_prompt_pack`` again."""
    from curation.test_pipeline import _FakeOpenSearch
    from src.routers.curation import pipeline
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': _body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': _body('B')}).status_code
        == 200
    )

    # What /start does at request time: resolve-and-pin once.
    name, revision = asyncio.run(resolve_run_prompt_pack(c.fake_os, 'my_pack@1'))
    assert (name, revision) == ('my_pack', 1)

    def _must_not_be_called(*_a, **_kw):  # pragma: no cover - failure path
        raise AssertionError(
            'pipeline_auto_label re-resolved a pinned prompt_pack instead of using the pin'
        )

    monkeypatch.setattr(pipeline, 'resolve_run_prompt_pack', _must_not_be_called)
    fake_os = _FakeOpenSearch({})
    summary = asyncio.run(
        pipeline._run_auto_label(
            opensearch=fake_os,
            train_clusters=False,
            promote_min_purity=0.85,
            promote_min_members=4,
            vlm_batch_size=32,
            vlm_concurrency=8,
            max_vlm_crops=0,
            classifier_confidence_skip_vlm=0.80,
            clustering_method=None,
            run_vlm=False,
            recluster_unvalidated=False,
            reassign_only=False,
            run_auto_promote=False,
            gate_max_rank=None,
            gate_min_blur_ratio=None,
            n_clusters=None,
            class_id=None,
            cluster_id=None,
            prompt_pack=name,
            prompt_pack_revision=revision,
            # R5-2 fix (W3/W4 round-5 review): "already resolved" is now
            # its own explicit flag, not inferred from `prompt_pack_
            # revision is not None` -- `/start` sets it unconditionally
            # (pinned, bare-name, AND omitted alike), matching what a
            # real trigger dict now carries.
            prompt_pack_resolved=True,
        )
    )
    print('pinned job summary ->', summary['prompt_pack'], summary['prompt_pack_revision'])
    assert summary['prompt_pack'] == 'my_pack'
    assert summary['prompt_pack_revision'] == 1


# ---- N2: tag does not leak into the default-path cache ----------------------
def test_r4_n2_tag_isolated_from_default_path(app_client):
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _default_pack_name, _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    c = app_client
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': _body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': _body('B')}).status_code
        == 200
    )
    assert (
        c.put(f'{PREFIX}/my_pack', json={'expected_revision': 2, 'body': _body('A')}).status_code
        == 200
    )
    name, rev = asyncio.run(resolve_run_prompt_pack(c.fake_os, 'my_pack'))
    lab = _get_vlm_labeler(name, rev)
    assert rev == 3
    assert prompt_pack_stamp(lab._pack, revision=rev) == 'my_pack@3'
    d = _get_vlm_labeler(asyncio.run(_default_pack_name(c.fake_os)))
    print('per-run', prompt_pack_stamp(lab._pack), 'default route', prompt_pack_stamp(d._pack))
    assert prompt_pack_stamp(d._pack) == 'my_pack@1'
    # per-run draft pin
    name, rev = asyncio.run(resolve_run_prompt_pack(c.fake_os, 'my_pack@3'))
    assert prompt_pack_stamp(_get_vlm_labeler(name, rev)._pack) == 'my_pack@3'


# ---- N5: pinned-copy 404 on the PROFILE axis --------------------------------
def test_r4_n5_profile_pinned_copy_missing(app_client, monkeypatch):
    from src.config import DetectionProfile
    from src.services.config_store import get_config_store
    from src.services.config_store.index import activate, config_doc_id, save_config
    from src.services.config_store.store import reset_config_stores
    from src.services.detection import profile_registry
    from src.services.detection.profile_registry import get_active_region_profile

    fake = app_client.fake_os
    profile_registry._reset_registry_for_tests()
    profile_registry.register_profile(DetectionProfile(name='envdef'), default=True)
    try:
        idx = get_config_store().index

        async def _seed():
            await save_config(
                fake,
                idx,
                kind='region_profile',
                name='rp',
                body={'max_regions_per_item': 1},
                expected_revision=None,
            )
            await activate(
                fake, idx, axis='detection_profile', name='rp', revision=1, expected_active=None
            )
            await save_config(
                fake,
                idx,
                kind='region_profile',
                name='rp',
                body={'max_regions_per_item': 7},
                expected_revision=1,
            )

        asyncio.run(_seed())
        fake._docs[idx].pop(config_doc_id('region_profile', 'rp', 1))
        reset_config_stores()
        asyncio.run(get_config_store().refresh(fake))
        served = get_active_region_profile()
        print('profile pinned 404 ->', served and (served.name, served.max_regions_per_item))
        assert served is not None
        assert served.max_regions_per_item != 7
    finally:
        profile_registry._reset_registry_for_tests()


# ---- settings bridge: re-activate after an explicit deactivation ------------
@pytest.mark.parametrize('axis', ['prompt_pack', 'detection_profile'])
def test_r4_settings_reactivate_after_off(app_client, monkeypatch, axis):
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
        assert c.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
        val = 'good' if axis == 'prompt_pack' else 'wheel'
        r1 = c.put(SETTINGS, json={'defaults': {axis: val}})
        r2 = c.put(SETTINGS, json={'defaults': {axis: None}})
        r3 = c.put(SETTINGS, json={'defaults': {axis: val}})
        print(axis, r1.status_code, r2.status_code, r3.status_code, r3.text[:120])
        assert (r1.status_code, r2.status_code, r3.status_code) == (200, 200, 200)
    finally:
        profile_registry._reset_registry_for_tests()


def test_r4_activate_route_after_deactivate_with_served_etag(app_client):
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': None}}).status_code == 200
    served = c.get(f'{PREFIX}/active').json()
    print('GET /active after off:', {k: served.get(k) for k in ('active', 'etag', 'previous')})
    r = c.post(f'{PREFIX}/good/activate', json={'expected_active': served.get('active')})
    print('activate with served active ->', r.status_code, r.text[:150])
    assert r.status_code == 200


def test_r4_legacy_settings_doc_prompt_pack_overrides_activation(app_client):
    """A pre-W2 settings doc still carrying defaults.prompt_pack (no startup
    migration strips it) vs. a pack activated through the store."""
    from src.clients.curation_opensearch import update_curation_settings
    from src.routers.curation.vlm import _default_pack_name, _get_vlm_labeler

    c = app_client
    assert c.post(PREFIX, json={'name': 'legacy', 'body': _body('LEGACY')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'good', 'body': _body('GOOD')}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    asyncio.run(update_curation_settings(c.fake_os, {'prompt_pack': 'legacy'}))
    shown = c.get(SETTINGS).json()['defaults'].get('prompt_pack')
    name = asyncio.run(_default_pack_name(c.fake_os))
    served = _get_vlm_labeler(name)._pack.class_system
    print('GET /settings shows', shown, '| VLM routes default name', name, '| serves', served)
    assert served == 'GOOD'
