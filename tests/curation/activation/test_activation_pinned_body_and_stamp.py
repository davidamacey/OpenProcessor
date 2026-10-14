"""An activated pack or profile serves the body of its activated revision and stamps
writes with that revision, however the revision is reached: default route, per-run pin,
rollback, or a cold or failing config store."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from curation.activation.constants import PREFIX, pack_body
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


pytestmark = pytest.mark.usefixtures('vlm_env')


def _activate_then_put(c):
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': pack_body()}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    r = c.put(f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': pack_body('R2 DRAFT')})
    assert r.status_code == 200, r.text


def test_vlm_route_default_serves_pinned_body_and_stamp(app_client):
    from src.routers.curation.vlm import _default_pack_name, _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _activate_then_put(app_client)
    name = asyncio.run(_default_pack_name(app_client.fake_os))
    lab = _get_vlm_labeler(name)
    print('route default', name, lab._pack.class_system[:20], prompt_pack_stamp(lab._pack))
    assert lab._pack.class_system == GENERIC_ITEM_PACK.class_system
    assert prompt_pack_stamp(lab._pack) == 'my_pack@1'


def test_pipeline_default_serves_pinned_body_and_stamp(app_client):
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, None))
    lab = _get_vlm_labeler(name, rev)
    print('pipeline default', name, rev, lab._pack.class_system[:20], prompt_pack_stamp(lab._pack))
    assert lab._pack.class_system == GENERIC_ITEM_PACK.class_system
    assert prompt_pack_stamp(lab._pack) == 'my_pack@1'


def test_per_run_pin_of_current_draft_is_stamped_with_its_own_revision(app_client):
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


def test_per_run_explicit_name_resolves_latest_revision(app_client):
    """§3.7: per-run `name` = latest saved revision."""
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, 'my_pack'))
    lab = _get_vlm_labeler(name, rev)
    print('per-run explicit my_pack ->', rev, lab._pack.class_system[:20])
    assert lab._pack.class_system == 'R2 DRAFT'


def test_per_run_pin_of_active_old_revision(app_client):
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, 'my_pack@1'))
    lab = _get_vlm_labeler(name, rev)
    assert (name, rev) == ('my_pack', 1)
    assert lab._pack.class_system == GENERIC_ITEM_PACK.class_system
    assert prompt_pack_stamp(lab._pack) == 'my_pack@1'


def test_per_run_pin_of_superseded_revision_is_unresolvable(app_client):
    """r1 activated, r2 activated, r3 PUT. my_pack@1 exists in history."""
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': pack_body('R1')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': pack_body('R2')}
        ).status_code
        == 200
    )
    r = c.post(
        f'{PREFIX}/my_pack/activate',
        json={'expected_active': {'name': 'my_pack', 'revision': 1}},
    )
    assert r.status_code == 200, r.text
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 2, 'body': pack_body('R3')}
        ).status_code
        == 200
    )
    # `my_pack` is at CURRENT revision 3, and the currently
    # ACTIVE revision is 2 (rev 1 was superseded by activating rev 2
    # over it). Per `resolve_run_prompt_pack`'s own contract, this
    # process only resolves a per-run pin against its process-cached
    # current doc plus the store's currently-activated-revision pin --
    # NOT a full historical lookup. So rev 1 -- a real revision that
    # once existed and was even once active, but is neither current nor
    # currently-active anymore -- is exactly as unresolvable as the
    # never-saved rev 9. Both must 422 with the same "not resolvable in
    # this process" wording.
    for far_past in ('my_pack@1', 'my_pack@9'):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(resolve_run_prompt_pack(c.fake_os, far_past))
        assert exc_info.value.status_code == 422
        assert 'not resolvable in this process' in exc_info.value.detail['error']


def test_warm_store_transient_fetch_error_keeps_last_good_body(app_client, monkeypatch):
    from src.services.config_store import get_config_store
    from src.services.labeling.vlm_prompts import active_prompt_pack, get_prompt_pack

    c = app_client
    _activate_then_put(c)
    store = get_config_store()
    asyncio.run(store.refresh(c.fake_os))
    assert active_prompt_pack().class_system == GENERIC_ITEM_PACK.class_system
    # another process bumps the revision (e.g. unrelated pack created)
    assert c.post(PREFIX, json={'name': 'other', 'body': pack_body('OTHER')}).status_code == 201
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


def test_missing_pinned_copy_does_not_serve_draft(app_client):
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


def test_vlm_api_default_labeler_serves_pinned_body(app_client):
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


def test_pipeline_omitted_prompt_pack_serves_pinned_body(app_client):
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _get_vlm_labeler

    _activate_then_put(app_client)
    name, rev = asyncio.run(resolve_run_prompt_pack(app_client.fake_os, None))
    labeler = _get_vlm_labeler(name, rev)
    print('pipeline default', name, rev, labeler._pack.class_system[:30])
    assert labeler._pack.class_system == GENERIC_ITEM_PACK.class_system


def test_pinned_fetch_failure_on_cold_reload_does_not_serve_put_body(app_client, monkeypatch):
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


def test_put_on_active_pack_creates_new_revision_and_keeps_active_revision(app_client):
    _activate_then_put(app_client)
    active = app_client.get(f'{PREFIX}/active').json()
    doc = app_client.get(f'{PREFIX}/my_pack').json()
    print(active, {k: doc.get(k) for k in ('revision', 'active', 'active_revision')})
    assert active['active']['revision'] == 1
    assert doc['revision'] == 2


def test_per_run_name_at_past_revision_labels_successfully(app_client):
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


def test_active_revision_pin_is_honored(app_client) -> None:
    from src.services.labeling.vlm_prompts import active_prompt_pack, prompt_pack_stamp

    assert app_client.post(PREFIX, json={'name': 'my_pack', 'body': pack_body()}).status_code == 201
    r = app_client.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None})
    assert r.status_code == 200, r.text
    body2 = pack_body()
    body2['class_system'] = 'EDITED AFTER ACTIVATION'
    r = app_client.put(f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': body2})
    assert r.status_code == 200, r.text
    active = app_client.get(f'{PREFIX}/active').json()
    served = active_prompt_pack()
    stamp = prompt_pack_stamp(served)
    print(
        'ACTIVE REF', active['active'], 'SERVED class_system', served.class_system, 'STAMP', stamp
    )
    # activation says revision 1 -> the served body must be revision 1's
    assert active['active']['revision'] == 1
    assert served.class_system == GENERIC_ITEM_PACK.class_system, (
        f'served rev-2 body while activation+stamp say rev 1 ({stamp})'
    )


def test_per_run_tag_isolated_from_default_path(app_client):
    from src.routers.curation.pipeline_params import resolve_run_prompt_pack
    from src.routers.curation.vlm import _default_pack_name, _get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    c = app_client
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': pack_body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': pack_body('B')}
        ).status_code
        == 200
    )
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 2, 'body': pack_body('A')}
        ).status_code
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


def test_profile_missing_pinned_copy_does_not_serve_draft(app_client, monkeypatch):
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
