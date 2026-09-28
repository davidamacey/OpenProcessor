"""Round-6 reviewer probes (scratch copy only). Each asserts CORRECT behavior;
a red test is a real defect."""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from typing import Any

import pytest

from curation.test_r4_probes import _body, _stripped
from curation.test_r5_probes import (  # noqa: F401 - fixtures
    PREFIX,
    RP,
    SETTINGS,
    _reset_caches,
    app_client,
    ready_state,
    registry,
)


def _live_pair():
    from src.services.detection.profile_registry import get_active_region_profile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    return active_prompt_pack(), get_active_region_profile()


# ---- 1. combined PUT: no over-rejection, and extra shapes -------------------
def _setup_packs(c):
    assert c.post(PREFIX, json={'name': 'good', 'body': _body('GOOD')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'single', 'body': _stripped()}).status_code == 201


@pytest.mark.parametrize('order', ['pack_first', 'profile_first'])
def test_r6_combined_reverse_pairing_now_valid_is_accepted(
    app_client,  # noqa: F811 - pytest fixture param shadows the cross-module import
    ready_state,  # noqa: F811 - pytest fixture param shadows the cross-module import
    registry,  # noqa: F811 - pytest fixture param shadows the cross-module import
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
    pack, prof = _live_pair()
    print(order, r.status_code, r.text[:200], pack.class_system, prof and prof.name)
    assert r.status_code == 200
    assert pack.class_system == 'GOOD'
    assert prof is not None
    assert prof.name == 'multi'


def test_r6_combined_stripped_with_profile_off_is_accepted(app_client, ready_state, registry):  # noqa: F811 - pytest fixture param shadows the cross-module import
    c = app_client
    _setup_packs(c)
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}}).status_code == 200
    r = c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single', 'detection_profile': 'off'}})
    pack, prof = _live_pair()
    print(r.status_code, r.text[:200], pack.class_system, prof)
    assert r.status_code == 200
    assert pack.class_system == 'STRIPPED'
    assert prof is None


def test_r6_combined_pack_off_with_multi_is_validated_against_env_default(
    app_client,  # noqa: F811 - pytest fixture param shadows the cross-module import
    ready_state,  # noqa: F811 - pytest fixture param shadows the cross-module import
    registry,  # noqa: F811 - pytest fixture param shadows the cross-module import
):
    """Pack -> None (env default GENERIC pack, has multi-box keys) + multi."""
    c = app_client
    _setup_packs(c)
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single'}}).status_code == 200
    r = c.put(SETTINGS, json={'defaults': {'prompt_pack': None, 'detection_profile': 'multi'}})
    pack, prof = _live_pair()
    print(r.status_code, r.text[:200], pack.name, pack.class_system, prof and prof.name)
    assert r.status_code == 200
    assert pack.class_system != 'STRIPPED'


def test_r6_combined_same_profile_already_live_still_rejects(app_client, ready_state, registry):  # noqa: F811 - pytest fixture param shadows the cross-module import
    c = app_client
    _setup_packs(c)
    assert c.put(SETTINGS, json={'defaults': {'prompt_pack': 'good'}}).status_code == 200
    assert c.put(SETTINGS, json={'defaults': {'detection_profile': 'multi'}}).status_code == 200
    r = c.put(SETTINGS, json={'defaults': {'prompt_pack': 'single', 'detection_profile': 'multi'}})
    pack, _prof = _live_pair()
    print(r.status_code, r.text[:200], pack.class_system)
    assert r.status_code == 422
    assert pack.class_system == 'GOOD'


# ---- 2. /start omitted: what the SEPARATE worker process actually does -----
def _start_args(c, monkeypatch, query):
    from src.services.curation.autolabel import job as auto_label_job

    captured: dict[str, Any] = {}

    def fake_start(fn, args):
        captured.update(args)
        return {'job_id': 'x'}

    monkeypatch.setattr(auto_label_job, 'start_job', fake_start)
    r = c.post(f'/curation/projects/default/pipeline/auto_label/start{query}')
    assert r.status_code in (200, 202), r.text
    return {k: v for k, v in captured.items() if k != 'opensearch'}


def _pack_setup(c):
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': _body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': _body('B DRAFT')}
        ).status_code
        == 200
    )


@pytest.mark.parametrize(('query', 'want'), [('', 'A'), ('?prompt_pack=my_pack@1', 'A')])
def test_r6_worker_process_cold_store(app_client, monkeypatch, query, want):  # noqa: F811 - pytest fixture param shadows the cross-module import
    """auto_label_worker is a separate container: its config store starts
    empty and nothing on the job path refreshes it once
    prompt_pack_resolved=True skips resolve_run_prompt_pack.

    Uses ``c.fake_os`` (not a fresh, unrelated fake) as the job's
    ``opensearch`` client -- this is the one piece of state that IS
    shared with a real cross-process worker (the same OpenSearch
    cluster); a genuinely different, empty backing store would never be
    able to resolve the pack regardless of any fix and would not be
    testing what R6-1 is about (a stale in-memory snapshot, not a
    disconnected client)."""
    from src.routers.curation import pipeline
    from src.routers.curation.vlm import _get_vlm_labeler
    from src.services.config_store import get_config_store
    from src.services.config_store.store import reset_config_stores

    c = app_client
    _pack_setup(c)
    args = _start_args(c, monkeypatch, query)
    reset_config_stores()  # a different process: cold store
    args['train_clusters'] = False
    args['run_vlm'] = False
    summary = asyncio.run(pipeline._run_auto_label(opensearch=c.fake_os, **args))
    print('store loaded after job stages:', get_config_store().current.loaded_at)
    # Exactly what `_run_auto_label`'s VLM stage does (R6-1b:
    # `prompt_pack_omitted` -- not the echoed `summary['prompt_pack']` --
    # decides whether the labeler resolves against `None`/`None`).
    omitted = args.get('prompt_pack_omitted', False)
    try:
        pack_arg = None if omitted else summary['prompt_pack']
        rev_arg = None if omitted else summary['prompt_pack_revision']
        lab = _get_vlm_labeler(pack_arg, rev_arg)
        got = lab._pack.class_system
    except Exception as exc:
        got = repr(exc)
    print(repr(query), args['prompt_pack'], args['prompt_pack_revision'], omitted, '->', got)
    assert got == want


def test_r6_omitted_job_after_active_switch_serves_draft(app_client, monkeypatch):  # noqa: F811 - pytest fixture param shadows the cross-module import
    """/start omitted (my_pack@1 active, r2 draft). Before the VLM stage the
    operator activates another pack. The job must serve either the pin
    (A) or the new active (OTHER), never my_pack's un-activated draft."""
    from src.routers.curation.vlm import _get_vlm_labeler

    c = app_client
    _pack_setup(c)
    assert c.post(PREFIX, json={'name': 'other', 'body': _body('OTHER')}).status_code == 201
    args = _start_args(c, monkeypatch, '')
    r = c.post(
        f'{PREFIX}/other/activate', json={'expected_active': {'name': 'my_pack', 'revision': 1}}
    )
    assert r.status_code == 200, r.text
    # R6-1b fix: the job resolves against `None`/`None` when
    # `prompt_pack_omitted` is set, exactly like `_run_auto_label`'s VLM
    # stage -- not the echoed `(prompt_pack, prompt_pack_revision)`.
    assert args['prompt_pack_omitted'] is True
    lab = _get_vlm_labeler(None, None)
    print(
        'trigger', args['prompt_pack'], args['prompt_pack_revision'], '->', lab._pack.class_system
    )
    assert lab._pack.class_system in ('A', 'OTHER')


def test_r6_hidden_resolved_flag_is_client_settable(app_client):  # noqa: F811 - pytest fixture param shadows the cross-module import
    """A direct POST /pipeline/auto_label caller can set the hidden flag and
    skip the unknown-pack 422."""
    c = app_client
    r = c.post(
        '/curation/projects/default/pipeline/auto_label',
        params={'prompt_pack': 'nope', 'prompt_pack_resolved': 'true', 'train_clusters': 'false'},
    )
    print('direct+resolved flag ->', r.status_code, r.text[:200])
    assert r.status_code == 422


# ---- 3. rollback to deleted (both axes) + m-a -------------------------------
def test_r6_rollback_deleted_pack_409_previous_deleted(app_client):  # noqa: F811 - pytest fixture param shadows the cross-module import
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


def test_r6_rollback_deleted_profile_409_previous_deleted(app_client, ready_state, registry):  # noqa: F811 - pytest fixture param shadows the cross-module import
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


def test_r6_rollback_refreshes_snapshot_before_gating(app_client, ready_state, registry):  # noqa: F811 - pytest fixture param shadows the cross-module import
    """m-a: another process activates 'multi' (direct index write, this
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
    pack, prof = _live_pair()
    print(rb.status_code, rb.text[:200], pack.class_system, prof and prof.name)
    assert rb.status_code == 422
    assert pack.class_system == 'GOOD'


# ---- 4. clone: pack-only target-context gap ---------------------------------
def _clone_pair(profile_activation: tuple[str | None, int | None] | None, plant_profile: bool):
    """Plant pack 'single' (STRIPPED) active in the source plus the given
    profile activation, clone into a fresh target, return the target's
    effective live (pack, profile)."""
    import os
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, MagicMock

    import src.config.curation as curation_mod
    from curation._fake_config_opensearch import FakeConfigOpenSearch
    from src.config.curation import base_curation_config
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
            source, target = _rec('r6alpha'), _rec('r6beta')
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
                pack, prof = _live_pair()
            return outcome, pack, prof
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            curation_mod._default_curation_config = None
            reset_config_stores()


@pytest.fixture
def multi_default_registry():
    from src.config import DetectionProfile
    from src.services.detection import profile_registry

    profile_registry._reset_registry_for_tests()
    profile_registry._ENV_RESOLVED = True
    profile_registry.register_profile(
        DetectionProfile(
            name='multi',
            detector_model='wheel_detector',
            text_reader='none',
            max_regions_per_item=3,
        ),
        default=True,
    )
    profile_registry.register_profile(
        DetectionProfile(
            name='base',
            detector_model='wheel_detector',
            text_reader='none',
            max_regions_per_item=1,
        )
    )
    yield profile_registry
    profile_registry._reset_registry_for_tests()


@pytest.mark.parametrize(
    'variant',
    ['source_profile_off', 'source_profile_registry_non_default'],
)
def test_r6_clone_pack_only_gap(ready_state, multi_default_registry, variant):  # noqa: F811 - pytest fixture param shadows the cross-module import
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


# ---- R6-m4 (3rd item): GET /active source label survives a deleted-but- ----
# still-activated name (m-b, W3/W4 round-5 review) -- no landed guard
# before this (mutation stayed green).
def test_r6_active_label_stored_for_deleted_but_still_activated_pack(
    app_client,  # noqa: F811 - pytest fixture param shadows the cross-module import
    ready_state,  # noqa: F811 - pytest fixture param shadows the cross-module import
    registry,  # noqa: F811 - pytest fixture param shadows the cross-module import
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
    assert c.post(PREFIX, json={'name': 'pack_a', 'body': _body('A')}).status_code == 201
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


# ---- R6-1b unit guard: the VLM-labeler resolution helper itself ------------
def test_r6_labeler_resolution_args_uses_none_when_omitted():
    """Direct unit test of `labeler_resolution_args` (R6-1b fix) -- the
    inline version of this check in `_run_auto_label` is otherwise only
    ever exercised indirectly (through `run_vlm=True` driving the full
    pipeline), which the round-6 probes above don't do. This is the
    guard that actually fails if the fix is reverted to always using the
    echoed `(prompt_pack, prompt_pack_revision)`."""
    from src.routers.curation.pipeline_params import labeler_resolution_args

    assert labeler_resolution_args('my_pack', 1, prompt_pack_omitted=True) == (None, None)
    assert labeler_resolution_args('my_pack', 1, prompt_pack_omitted=False) == ('my_pack', 1)
    assert labeler_resolution_args(None, None, prompt_pack_omitted=False) == (None, None)
