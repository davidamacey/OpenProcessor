"""Cloning project settings through the real flow (validate, then apply) against an
in-memory config index, with no artificial sleep: the activated body is what lands in the
target, drafts are never promoted, and a conflicting target is refused before any write."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from projects.clone_flow_fakes import (  # noqa: F401 - isolated_state is an autouse fixture
    Fake,
    activate_config,
    clone_into,
    isolated_state,
    patch_lifecycle,
    record,
    save,
    target_docs,
)
from src.config.project_context import bind_project
from src.core.exceptions import RequestRejectedError
from src.services.projects.clone import _apply_clone, _validate_clone


pytestmark = pytest.mark.unbound


async def _target_state(client, target):
    from src.config import get_curation_config
    from src.services.config_store.index import config_doc_id, get_activation

    with bind_project(target):
        idx = get_curation_config().configs_index
        act = await get_activation(client, idx, 'prompt_pack')
        cur = None
        active_body = None
        if act and act.get('name'):
            cur = (await client.get(index=idx, id=config_doc_id('prompt_pack', act['name'])))[
                '_source'
            ]
            active_body = (
                await client.get(
                    index=idx, id=config_doc_id('prompt_pack', act['name'], act['revision'])
                )
            )['_source']['body']
        settings = client._docs.get(idx, {})
    return act, cur, active_body, settings


async def _clone(client, source, target, monkeypatch, axes=None):
    from src.services.projects import lifecycle as lifecycle_mod

    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=source))
    return await lifecycle_mod.clone_settings(
        client, target_record=target, from_slug=source.slug, axes=axes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('axes', [None, ['prompt_packs', 'activations']])
async def test_clone_settings_flow_copies_the_activated_body(monkeypatch, axes):
    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    await _clone(client, source, target, monkeypatch, axes=axes)
    act, _cur, active_body, _ = await _target_state(client, target)
    print('target act', act and (act['name'], act['revision']), 'body', active_body)
    assert active_body == {'class_system': 'R1'}


@pytest.mark.asyncio
async def test_clone_settings_does_not_promote_a_source_draft(monkeypatch):
    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R2 DRAFT'}, expected=1)
    await _clone(client, source, target, monkeypatch, axes=None)
    act, cur, active_body, _ = await _target_state(client, target)
    print(
        'draft clone: target active',
        (act['name'], act['revision']),
        active_body,
        '| target current doc',
        cur['revision'],
        cur['body'],
    )
    assert active_body == {'class_system': 'R1'}


@pytest.mark.asyncio
async def test_draft_only_source_pack_is_never_active_in_target(monkeypatch):
    """Source has a pack that was PUT but never activated at all."""
    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'never_active', {'class_system': 'D'})
    await _clone(client, source, target, monkeypatch, axes=None)
    act, *_ = await _target_state(client, target)
    print('draft-only source -> target activation', act)
    assert not (act and act.get('name'))


# ---- existing-target (clone_settings_into-shaped) residual gaps ---------------
@pytest.mark.asyncio
async def test_existing_target_with_same_named_unactivated_profile_conflicts_before_writes(
    monkeypatch,
):
    """Existing target has an un-activated stored region profile with the same
    name as the source's ACTIVE profile. Expect: clean 409 BEFORE any write."""
    from src.config import get_curation_config

    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    await save(client, source, 'region_profile', 'badge', {'x': 1})
    await activate_config(client, source, 'detection_profile', 'badge', 1)
    await save(client, target, 'region_profile', 'badge', {'x': 'TARGET OWN'})
    with bind_project(target):
        tidx = get_curation_config().configs_index
    before = set(client._docs.get(tidx, {}))
    try:
        await _clone(client, source, target, monkeypatch, axes=None)
        outcome = 'ok'
    except RequestRejectedError as exc:
        outcome = f'{exc.status_code} {exc.detail}'
    after = set(client._docs.get(tidx, {}))
    print('profile-collision clone ->', outcome, '| new target docs:', sorted(after - before))
    assert not (outcome != 'ok' and after - before), 'partial write before 409'


@pytest.mark.asyncio
async def test_existing_target_previously_deactivated_accepts_clone(monkeypatch):
    """Existing target once activated a pack then deactivated it ('off').
    _validate_clone treats that as 'no activation'; activate(expected_active=None)
    then conflicts."""
    from src.config import get_curation_config

    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    # target: activate a built-in/env id then turn it off (no stored packs left)
    await activate_config(client, target, 'prompt_pack', 'generic_item', None)
    await activate_config(
        client,
        target,
        'prompt_pack',
        None,
        None,
        expected={'name': 'generic_item', 'revision': None},
    )
    with bind_project(target):
        tidx = get_curation_config().configs_index
    before = set(client._docs.get(tidx, {}))
    try:
        await _clone(client, source, target, monkeypatch, axes=None)
        outcome = 'ok'
    except RequestRejectedError as exc:
        outcome = f'{exc.status_code} {exc.detail}'
    after = set(client._docs.get(tidx, {}))
    print('deactivated-target clone ->', outcome, '| new target docs:', sorted(after - before))
    # A target that once deactivated this axis is genuinely EMPTY (no stored pack of its
    # own on this axis), so a clone into it must actually SUCCEED. Accepting a clean 409
    # (no docs written, nothing cloned) would mask a spurious `active_conflict` from
    # treating 'off' as `expected_active=None`.
    assert outcome == 'ok', f'clone into a previously-deactivated target must succeed: {outcome}'


@pytest.mark.asyncio
async def test_clone_into_a_target_with_a_stored_unactivated_pack_of_the_same_name_writes_nothing(
    monkeypatch,
):
    """The refusal comes from validation, before any write: a same-named
    stored pack that was never activated is the target's own history."""
    from src.config import get_curation_config

    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    await save(client, target, 'prompt_pack', 'wheel', {'class_system': 'MINE'})
    with bind_project(target):
        tidx = get_curation_config().configs_index
    before = {k: dict(v) for k, v in client._docs.get(tidx, {}).items()}
    with pytest.raises(RequestRejectedError) as exc:
        await _clone(client, source, target, monkeypatch, axes=['settings_defaults', 'activations'])
    assert exc.value.status_code == 409
    assert exc.value.detail['error'] == 'target_not_empty'
    assert client._docs.get(tidx, {}) == before


@pytest.mark.asyncio
async def test_clone_into_previously_deactivated_target_succeeds(monkeypatch):
    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    await activate_config(client, target, 'prompt_pack', 'generic_item', None)
    await activate_config(
        client,
        target,
        'prompt_pack',
        None,
        None,
        expected={'name': 'generic_item', 'revision': None},
    )
    patch_lifecycle(monkeypatch, source, target)
    await clone_into(client, source, target)
    from src.config import get_curation_config
    from src.services.config_store.index import get_activation

    with bind_project(target):
        act = await get_activation(client, get_curation_config().configs_index, 'prompt_pack')
    print('deactivated target ->', act)
    assert act is not None
    assert act['name'] == 'wheel'
    assert act['previous'] == {'name': None, 'revision': None}


@pytest.mark.parametrize(
    ('kind', 'axis', 'name'),
    [
        ('region_profile', 'detection_profile', 'badge'),
        ('prompt_pack', 'prompt_pack', 'wheel'),
    ],
)
@pytest.mark.asyncio
async def test_clone_into_same_named_unactivated_config_409_before_writes(
    monkeypatch, kind, axis, name
):
    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, kind, name, {'x': 1})
    await activate_config(client, source, axis, name, 1)
    await save(client, target, kind, name, {'x': 'TARGET OWN'})
    patch_lifecycle(monkeypatch, source, target)
    before = target_docs(client, target)
    with pytest.raises(RequestRejectedError) as ei:
        await clone_into(client, source, target)
    after = target_docs(client, target)
    changed = sorted(k for k in set(after) | set(before) if after.get(k) != before.get(k))
    print(kind, '->', ei.value.status_code, ei.value.detail, '| changed:', changed)
    assert ei.value.status_code == 409
    assert changed == []


@pytest.mark.asyncio
async def test_clone_into_target_with_stale_cache_does_not_partially_write(monkeypatch):
    """Another worker process stored 'badge' in the target <1s after this
    process last warmed the target store. Validation reads the warm cache."""
    from src.config import get_curation_config
    from src.services.config_store import get_config_store
    from src.services.config_store.index import save_config

    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'region_profile', 'badge', {'x': 1})
    await activate_config(client, source, 'detection_profile', 'badge', 1)
    with bind_project(target):
        store = get_config_store()
        await store.refresh(client)  # warm, empty
        # "other process" write: straight to the index, no apply_local here
        await save_config(
            client,
            get_curation_config().configs_index,
            kind='region_profile',
            name='badge',
            body={'x': 'OTHER WORKER'},
            expected_revision=None,
        )
    patch_lifecycle(monkeypatch, source, target)
    before = target_docs(client, target)
    try:
        await clone_into(client, source, target)
        outcome = 'ok'
    except RequestRejectedError as exc:
        outcome = f'{exc.status_code} {exc.detail}'
    after = target_docs(client, target)
    changed = sorted(k for k in set(after) | set(before) if after.get(k) != before.get(k))
    print('stale-cache race ->', outcome, '| changed:', changed)
    assert outcome == 'ok' or changed == [], 'partial write before 409'


@pytest.mark.asyncio
async def test_prompt_packs_plus_activations_axes_together() -> None:
    client = Fake()
    source = record('alpha')
    target = record('beta')
    from src.services.config_store.index import activate, save_config

    with bind_project(source):
        from src.config import get_curation_config

        idx = get_curation_config().configs_index
        doc = await save_config(
            client,
            idx,
            kind='prompt_pack',
            name='wheel',
            body={'class_system': 'x'},
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='prompt_pack',
            name='wheel',
            revision=doc['revision'],
            expected_active=None,
        )
    await _apply_clone(
        client,
        target_record=target,
        source=source,
        axes=['prompt_packs', 'activations'],
        target_activations=None,
    )


async def _seed(client, source, *, edit_after_activate):
    from src.config import get_curation_config
    from src.services.config_store.index import activate, save_config

    with bind_project(source):
        idx = get_curation_config().configs_index
        doc = await save_config(
            client,
            idx,
            kind='prompt_pack',
            name='wheel',
            body={'class_system': 'R1'},
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='prompt_pack',
            name='wheel',
            revision=doc['revision'],
            expected_active=None,
        )
        if edit_after_activate:
            await save_config(
                client,
                idx,
                kind='prompt_pack',
                name='wheel',
                body={'class_system': 'R2 NOT ACTIVATED'},
                expected_revision=1,
            )


@pytest.mark.asyncio
async def test_validate_then_apply_succeeds_without_waiting_for_cache_ttl(monkeypatch):
    """clone_settings / clone_settings_into run _validate_clone (which warms the
    target store) immediately before _apply_clone."""
    from src.services.projects import lifecycle as lifecycle_mod

    client = Fake()
    source, target = record('alpha'), record('beta')
    await _seed(client, source, edit_after_activate=False)
    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=source))
    src_rec, axes, target_activations = await _validate_clone(
        client, target_record=target, from_slug='alpha', axes=['prompt_packs', 'activations']
    )
    await _apply_clone(
        client,
        target_record=target,
        source=src_rec,
        axes=axes,
        target_activations=target_activations,
    )


@pytest.mark.asyncio
async def test_target_activates_the_source_activated_body_not_current():
    from src.config import get_curation_config
    from src.services.config_store.index import get_activation

    client = Fake()
    source, target = record('alpha'), record('beta')
    await _seed(client, source, edit_after_activate=True)
    await _apply_clone(
        client,
        target_record=target,
        source=source,
        axes=['prompt_packs', 'activations'],
        target_activations=None,
    )
    with bind_project(target):
        idx = get_curation_config().configs_index
        act = await get_activation(client, idx, 'prompt_pack')
        assert act is not None
        from src.services.config_store.index import config_doc_id

        doc = await client.get(
            index=idx, id=config_doc_id('prompt_pack', act['name'], act['revision'])
        )
    assert doc is not None
    print('target active', act['name'], act['revision'], doc['_source']['body'])
    assert doc['_source']['body'] == {'class_system': 'R1'}, (
        'target activated the source CURRENT (never-activated) body, not the activated revision'
    )


@pytest.mark.asyncio
async def test_validate_then_apply_after_cache_ttl(monkeypatch):
    import asyncio

    from src.services.projects import lifecycle as lifecycle_mod

    client = Fake()
    source, target = record('alpha'), record('beta')
    await _seed(client, source, edit_after_activate=False)
    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=source))
    src_rec, axes, target_activations = await _validate_clone(
        client, target_record=target, from_slug='alpha', axes=['prompt_packs', 'activations']
    )
    await asyncio.sleep(1.1)
    await _apply_clone(
        client,
        target_record=target,
        source=src_rec,
        axes=axes,
        target_activations=target_activations,
    )
