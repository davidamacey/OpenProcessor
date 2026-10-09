"""Round-5 reviewer probes (scratch copy only). Each asserts CORRECT behavior;
a red test is a real defect."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.test_r4_probes import _body, _stripped
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK  # noqa: F401


pytestmark = pytest.mark.usefixtures('vlm_env')


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.clients.curation_opensearch import settings_doc
    from src.services.config_store.store import reset_config_stores

    settings_doc._settings_cache.clear()
    reset_config_stores()
    yield
    settings_doc._settings_cache.clear()
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
RP = '/curation/projects/default/region_profiles'
SETTINGS = '/curation/projects/default/settings'


@pytest.fixture
def ready_state(monkeypatch):
    from src.services.triton_control import TritonControlService

    state = {'ready': {'wheel_detector', 'other_detector'}}

    async def _repo():
        return [{'name': n, 'state': 'READY', 'version': '1'} for n in state['ready']]

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())
    return state


@pytest.fixture
def registry():
    from src.config import DetectionProfile
    from src.services.detection import profile_registry

    profile_registry._reset_registry_for_tests()
    profile_registry.register_profile(
        DetectionProfile(
            name='base',
            detector_model='wheel_detector',
            text_reader='none',
            max_regions_per_item=1,
        ),
        default=True,
    )
    profile_registry.register_profile(
        DetectionProfile(
            name='multi',
            detector_model='wheel_detector',
            text_reader='none',
            max_regions_per_item=3,
        )
    )
    profile_registry.register_profile(
        DetectionProfile(name='other', detector_model='other_detector', text_reader='none')
    )
    yield profile_registry
    profile_registry._reset_registry_for_tests()


# ---- 1. Two-axis PUT /settings: each axis gated against the OTHER axis's
#         OLD state, not the state the same request is about to write -------
@pytest.mark.parametrize('order', ['pack_first', 'profile_first'])
def test_r5_two_axis_put_cannot_pair_stripped_pack_with_multi_profile(
    app_client, ready_state, registry, order
):
    from src.services.detection.profile_registry import get_active_region_profile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': _body('GOOD')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'single', 'body': _stripped()}).status_code == 201
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


def test_r5_control_single_axis_activate_refuses_stripped_under_multi(
    app_client, ready_state, registry
):
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': _body('GOOD')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'single', 'body': _stripped()}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}}).status_code == 200
    r = c.post(
        f'{PREFIX}/single/activate',
        json={'expected_active': {'name': 'good', 'revision': 1}, 'force': True},
    )
    assert r.status_code == 422


# ---- 2. Profile-axis rollback gate (no landed test for this axis) -----------
def test_r5_profile_rollback_refuses_profile_whose_detector_is_gone(
    app_client, ready_state, registry
):
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


# ---- 3. Client fallback: rollback to a superseded-but-real revision ---------
def test_r5_rollback_to_superseded_revision_serves_pinned_body(app_client):
    from src.services.labeling.vlm_prompts import active_prompt_pack, prompt_pack_stamp

    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': _body('A1')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'pack_b', 'body': _body('B1')}).status_code == 201
    assert c.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/pack_a', json={'expected_revision': 1, 'body': _body('A2 DRAFT')}
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


def test_r5_rollback_superseded_revision_that_fails_gate_now_422s(app_client, monkeypatch):
    """The fallback must not let a superseded revision skip validation:
    pack_a@1 is STRIPPED, pack_a current r2 is GOOD. After the profile goes
    multi-region, rolling back to pack_a@1 must 422 even though the CURRENT
    pack_a (r2) would pass."""
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': _stripped()}).status_code == 201
    assert c.post(PREFIX, json={'name': 'pack_b', 'body': _body('B1')}).status_code == 201
    assert c.post(f'{PREFIX}/pack_a/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/pack_a', json={'expected_revision': 1, 'body': _body('A2 GOOD')}
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


def test_r5_rollback_to_deleted_pack(app_client):
    """Record-and-assert: a pack deleted while it is the activation's
    `previous`. Before R4-1 the gate did not exist; with the client
    fallback the deleted pack's @rev copy is found, so rollback resurrects
    a pack that no longer exists in the list."""
    from src.services.labeling.vlm_prompts import active_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': _body('A1')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'pack_b', 'body': _body('B1')}).status_code == 201
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


# ---- 4. R4-2: X -> none -> Y (different target) on both axes ----------------
@pytest.mark.parametrize('axis', ['prompt_pack', 'detection_profile'])
def test_r5_settings_x_none_y(app_client, ready_state, registry, axis):
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
    assert c.post(PREFIX, json={'name': 'good2', 'body': _body('G2')}).status_code == 201
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


def test_r5_settings_after_deactivate_route(app_client, ready_state, registry):
    c = app_client
    assert c.post(f'{RP}/base/activate', json={'expected_active': None}).status_code == 200
    d = c.post(f'{RP}/deactivate', json={'expected_active': {'name': 'base', 'revision': None}})
    assert d.status_code == 200, d.text
    r = c.put(SETTINGS, json={'defaults': {'detection_profile': 'other'}})
    print('after /deactivate ->', r.status_code, r.text[:160])
    assert r.status_code == 200


def test_r5_settings_on_cold_store_after_off(app_client, ready_state, registry):
    """Another process deactivated; this process's store is reset (cold)."""
    from src.services.config_store.store import reset_config_stores

    c = app_client
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': None}}).status_code == 200
    assert c.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': None}}).status_code == 200
    reset_config_stores()
    r = c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}})
    print('cold after off ->', r.status_code, r.text[:160])
    assert r.status_code == 200


# ---- 5. R4-3: /start with the pack OMITTED (the Cropwright default path) ----
def _run_job(c, monkeypatch, query):
    from curation.test_pipeline import _FakeOpenSearch
    from src.routers.curation import pipeline
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack as real_resolve
    from src.services.curation.autolabel import job as auto_label_job

    captured = {}

    def fake_start(fn, args):
        captured.update(args)
        return {'job_id': 'x'}

    monkeypatch.setattr(auto_label_job, 'start_job', fake_start)
    r = c.post(f'/curation/projects/default/pipeline/auto_label/start{query}')
    assert r.status_code in (200, 202), r.text

    async def _resolve(_os, pp):
        return await real_resolve(c.fake_os, pp)

    monkeypatch.setattr(pipeline, 'resolve_run_prompt_pack', _resolve)
    args = {k: v for k, v in captured.items() if k != 'opensearch'}
    args['train_clusters'] = False
    args['run_vlm'] = False
    summary = asyncio.run(pipeline._run_auto_label(opensearch=_FakeOpenSearch({}), **args))
    return captured, summary


@pytest.mark.parametrize(
    ('query', 'want_rev'),
    [('', 1), ('?prompt_pack=my_pack@1', 1), ('?prompt_pack=my_pack', 2)],
)
def test_r5_start_job_runs_what_the_request_resolved(app_client, monkeypatch, query, want_rev):
    from src.routers.curation.vlm import _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    c = app_client
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': _body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': _body('B DRAFT')}
        ).status_code
        == 200
    )
    captured, summary = _run_job(c, monkeypatch, query)
    lab = _get_vlm_labeler(summary['prompt_pack'], summary['prompt_pack_revision'])
    stamp = prompt_pack_stamp(lab._pack, revision=summary['prompt_pack_revision'])
    print(
        repr(query),
        'trigger',
        (captured['prompt_pack'], captured['prompt_pack_revision']),
        '| job',
        (summary['prompt_pack'], summary['prompt_pack_revision']),
        '| serves',
        lab._pack.class_system,
        stamp,
    )
    want_body = 'A' if want_rev == 1 else 'B DRAFT'
    assert lab._pack.class_system == want_body
    assert stamp == f'my_pack@{want_rev}'


# ---- 6. Walk EVERY real activation-writer, one gate-failing revision -------
# (asked for in round 4, still missing per round 5's review). Enumerates the
# real call sites -- direct activate x2, settings-bridge x2 axes + the
# combined-request case (Blocker R5-1), rollback x2, and clone (Major R5-3)
# -- against the SAME never-bypassable failure (a multi-box-stripped pack
# paired with a multi-region profile), so a future caller that skips the
# gate on ANY of these paths turns this test red.
def _active_ref(c: Any, kind: str) -> dict[str, Any]:
    """Current ``{'name':..., 'revision':...}`` for ``prompt_packs`` or
    ``region_profiles``, straight from the router -- avoids guessing OCC
    state (an 'off' deactivation stores ``{'name': None, 'revision':
    None}``, not Python ``None``, R4-2) across a long scripted sequence."""
    active = c.get(f'/curation/projects/default/{kind}/active').json()['active']
    return {'name': active.get('name'), 'revision': active.get('revision')}


def test_r5_walk_every_activation_writer_rejects_gate_failing_pair(
    app_client, ready_state, registry
):
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': _body('GOOD')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'single', 'body': _stripped()}).status_code == 201
    assert c.post(f'{PREFIX}/good/activate', json={'expected_active': None}).status_code == 200
    results: dict[str, int] = {}

    # (a) direct activate: prompt_pack, paired against an already-active
    # multi-region profile.
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}}).status_code == 200
    r = c.post(
        f'{PREFIX}/single/activate',
        json={'expected_active': _active_ref(c, 'prompt_packs'), 'force': True},
    )
    results['activate_pack'] = r.status_code
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': None}}).status_code == 200

    # (b) direct activate: detection_profile, paired against an
    # already-active stripped pack.
    assert (
        c.post(
            f'{PREFIX}/single/activate',
            json={'expected_active': _active_ref(c, 'prompt_packs'), 'force': True},
        ).status_code
        == 200
    )
    r = c.post(f'{RP}/multi/activate', json={'expected_active': _active_ref(c, 'region_profiles')})
    results['activate_profile'] = r.status_code
    assert (
        c.post(
            f'{PREFIX}/good/activate',
            json={'expected_active': _active_ref(c, 'prompt_packs'), 'force': True},
        ).status_code
        == 200
    )

    # (c) settings bridge, prompt_pack axis alone.
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}}).status_code == 200
    r = c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single'}})
    results['settings_pack'] = r.status_code
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': None}}).status_code == 200

    # (d) settings bridge, detection_profile axis alone.
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single'}}).status_code == 200
    r = c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}})
    results['settings_profile'] = r.status_code
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}}).status_code == 200

    # (e) settings bridge, BOTH axes in one request (Blocker R5-1).
    r = c.put(
        SETTINGS,
        json={'defaults': {'prompt_pack': 'single', 'detection_profile': 'multi'}},
    )
    from src.services.detection.profile_registry import get_active_region_profile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    served_pack, served_prof = active_prompt_pack(), get_active_region_profile()
    combined_pair_live = (
        served_pack.class_system == 'STRIPPED'
        and served_prof is not None
        and served_prof.max_regions_per_item > 1
    )
    # R6-m2 fix (W3/W4 round-6 review): the old `409 if combined_pair_live
    # else 422` bucketed BOTH outcomes into the shared `(409, 422)` loop
    # assertion below, so this entry passed even when the gate was
    # disabled entirely -- the round-5 mutation proved it stays green
    # while `test_r5_two_axis_put...` (the dedicated R5-1 test) correctly
    # goes red. Assert directly on what actually matters: the request was
    # rejected (422 specifically -- this route's own gate-failure code,
    # not some unrelated `active_conflict` 409) AND the bad pairing never
    # went live. Not folded into `results`/the shared loop below, so a
    # future change to that loop's acceptance set can't silently make
    # this vacuous again.
    assert r.status_code == 422, f'combined PUT did not reject the bad pairing: {r.status_code}'
    assert not combined_pair_live, 'combined PUT let the gate-failing pack+profile pair go live'

    # Reset to a clean, valid baseline before the rollback cases.
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': None}}).status_code == 200

    # (f) rollback: prompt_pack. previous = stripped 'single'@1, current
    # (live) profile = multi.
    assert (
        c.post(
            f'{PREFIX}/single/activate',
            json={'expected_active': _active_ref(c, 'prompt_packs'), 'force': True},
        ).status_code
        == 200
    )
    good_before_rollback = _active_ref(c, 'prompt_packs')
    assert (
        c.post(
            f'{PREFIX}/good/activate', json={'expected_active': good_before_rollback, 'force': True}
        ).status_code
        == 200
    )
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}}).status_code == 200
    r = c.post(
        f'{PREFIX}/active/rollback', json={'expected_active': _active_ref(c, 'prompt_packs')}
    )
    results['rollback_pack'] = r.status_code
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': None}}).status_code == 200

    # (g) rollback: detection_profile. previous = multi, current (live)
    # pack = stripped 'single'.
    assert (
        c.post(
            f'{RP}/base/activate', json={'expected_active': _active_ref(c, 'region_profiles')}
        ).status_code
        == 200
    )
    assert (
        c.post(
            f'{RP}/multi/activate', json={'expected_active': _active_ref(c, 'region_profiles')}
        ).status_code
        == 200
    )
    base_before_rollback = _active_ref(c, 'region_profiles')
    assert (
        c.post(f'{RP}/base/activate', json={'expected_active': base_before_rollback}).status_code
        == 200
    )
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single'}}).status_code == 200
    r = c.post(f'{RP}/active/rollback', json={'expected_active': _active_ref(c, 'region_profiles')})
    results['rollback_profile'] = r.status_code

    print('walk results:', results)
    for writer, status in results.items():
        assert status in (409, 422), f'{writer} did not reject the gate-failing pair: {status}'

    # (h) project clone: source has the SAME gate-failing pair (pack
    # 'single' STRIPPED + profile 'multi' multi-region) planted directly
    # as its active pair -- bypassing the gate the way a pre-fix build of
    # this codebase could have -- and clone must not carry it into a
    # fresh target as live.
    _assert_clone_rejects_gate_failing_pair()


def _assert_clone_rejects_gate_failing_pair() -> None:
    import tempfile
    from datetime import UTC, datetime

    from src.config.curation import base_curation_config
    from src.config.project_context import bind_project
    from src.config.projects import ProjectRecord, resources_for_new
    from src.services.config_store.index import activate, get_activation, save_config
    from src.services.config_store.store import reset_config_stores

    class _Fake(FakeConfigOpenSearch):
        async def count(self, index, body=None):  # noqa: ARG002
            return {'count': 0}

        async def bulk(self, body, refresh=False):  # noqa: ARG002
            return {'errors': False, 'items': []}

    def _rec(slug: str) -> Any:
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
        import os

        old_state, old_data = (
            os.environ.get('OP_STATE_DIR'),
            os.environ.get('OP_PROJECTS_DATA_ROOT'),
        )
        os.environ['OP_STATE_DIR'] = f'{tmp}/state'
        os.environ['OP_PROJECTS_DATA_ROOT'] = f'{tmp}/projects_data'
        import src.config.curation as curation_mod

        curation_mod._default_curation_config = None
        reset_config_stores()
        try:
            client = _Fake()
            source, target = _rec('r5walk_alpha'), _rec('r5walk_beta')

            with bind_project(source):
                from src.config import get_curation_config

                idx = get_curation_config().configs_index
                asyncio.run(
                    save_config(
                        client,
                        idx,
                        kind='prompt_pack',
                        name='single',
                        body=_stripped(),
                        expected_revision=None,
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
                profile_body = {
                    'detector_model': 'wheel_detector',
                    'text_reader': 'none',
                    'max_regions_per_item': 3,
                }
                asyncio.run(
                    save_config(
                        client,
                        idx,
                        kind='region_profile',
                        name='multi',
                        body=profile_body,
                        expected_revision=None,
                    )
                )
                asyncio.run(
                    activate(
                        client,
                        idx,
                        axis='detection_profile',
                        name='multi',
                        revision=1,
                        expected_active=None,
                    )
                )

            from unittest.mock import AsyncMock, MagicMock

            from src.services.projects import lifecycle as lifecycle_mod, registry as registry_mod
            from src.services.projects.clone import clone_settings_into

            orig_resolve = lifecycle_mod._resolve_existing
            orig_mutable = lifecycle_mod._get_mutable_record
            orig_write = lifecycle_mod.write_record
            orig_get_registry = registry_mod.get_project_registry
            lifecycle_mod._resolve_existing = AsyncMock(return_value=source)
            lifecycle_mod._get_mutable_record = AsyncMock(return_value=(target, 1, 1))
            lifecycle_mod.write_record = AsyncMock(return_value=None)
            reg = MagicMock()
            reg.ensure_fresh = AsyncMock(return_value=None)
            registry_mod.get_project_registry = lambda: reg
            try:
                with contextlib.suppress(Exception):
                    asyncio.run(
                        clone_settings_into(
                            client,
                            slug=target.slug,
                            from_slug=source.slug,
                            axes=None,
                            expected_revision=1,
                        )
                    )
            finally:
                lifecycle_mod._resolve_existing = orig_resolve
                lifecycle_mod._get_mutable_record = orig_mutable
                lifecycle_mod.write_record = orig_write
                registry_mod.get_project_registry = orig_get_registry

            with bind_project(target):
                from src.config import get_curation_config as _tgt_cfg

                tgt_idx = _tgt_cfg().configs_index
                pack_act = asyncio.run(get_activation(client, tgt_idx, 'prompt_pack'))
                prof_act = asyncio.run(get_activation(client, tgt_idx, 'detection_profile'))
            print('clone target activations:', pack_act, prof_act)
            assert not (
                (pack_act or {}).get('name') == 'single' and (prof_act or {}).get('name') == 'multi'
            ), 'clone carried the gate-failing pack+profile pair into the target as live'
        finally:
            if old_state is None:
                os.environ.pop('OP_STATE_DIR', None)
            else:
                os.environ['OP_STATE_DIR'] = old_state
            if old_data is None:
                os.environ.pop('OP_PROJECTS_DATA_ROOT', None)
            else:
                os.environ['OP_PROJECTS_DATA_ROOT'] = old_data
            curation_mod._default_curation_config = None
            reset_config_stores()
