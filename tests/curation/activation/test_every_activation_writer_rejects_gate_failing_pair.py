"""Every real activation writer rejects the same gate-failing pair (a multi-box-stripped
pack with a multi-region profile): direct activate, settings bridge, rollback and clone.
A new writer that skips the gate turns this test red."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.activation.constants import PREFIX, RP, SETTINGS, pack_body, stripped_pack_body


pytestmark = pytest.mark.usefixtures('vlm_env')


def _active_ref(c: Any, kind: str) -> dict[str, Any]:
    """Current ``{'name':..., 'revision':...}`` for ``prompt_packs`` or
    ``region_profiles``, straight from the router -- avoids guessing OCC
    state (an 'off' deactivation stores ``{'name': None, 'revision':
    None}``, not Python ``None``) across a long scripted sequence."""
    active = c.get(f'/curation/projects/default/{kind}/active').json()['active']
    return {'name': active.get('name'), 'revision': active.get('revision')}


def test_every_activation_writer_rejects_gate_failing_pair(app_client, ready_state, registry):
    c = app_client
    assert c.post(PREFIX, json={'name': 'good', 'body': pack_body('GOOD')}).status_code == 201
    assert c.post(PREFIX, json={'name': 'single', 'body': stripped_pack_body()}).status_code == 201
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

    # (e) settings bridge, BOTH axes in one request.
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
    # Not `409 if combined_pair_live else 422` folded into the shared `(409, 422)` loop
    # assertion below: that passed even when the gate was disabled entirely, while
    # `test_two_axis_put_cannot_pair_stripped_pack_with_multi_profile` correctly went
    # red. Assert directly on what actually matters: the request was
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
    # as its active pair -- bypassing the gate the way an older build could have --
    # and clone must not carry it into a fresh target as live.
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
            source, target = _rec('walk_alpha'), _rec('walk_beta')

            with bind_project(source):
                from src.config import get_curation_config

                idx = get_curation_config().configs_index
                asyncio.run(
                    save_config(
                        client,
                        idx,
                        kind='prompt_pack',
                        name='single',
                        body=stripped_pack_body(),
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
