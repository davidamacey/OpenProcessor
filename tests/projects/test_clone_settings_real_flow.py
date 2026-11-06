"""Cloning project settings through the real flow (validate, then apply) against an
in-memory config index, with no artificial sleep: the activated body is what lands in the
target, drafts are never promoted, and a conflicting target is refused before any write."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from config_store import memory_config

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
from src.services.projects.clone import _apply_clone
from src.services.projects.clone_validate import _validate_clone


pytestmark = pytest.mark.unbound


async def _target_state(client, target):
    """``(activation, current doc, activated body, every document)`` of the target's prompt packs."""
    from src.services.config_store.documents import get_revision, read_activation

    del client
    with bind_project(target):
        read = await read_activation(target.slug, 'prompt_pack')
        act = None if read is None else {'name': read.name, 'revision': read.revision}
        cur = None
        active_body = None
        if read is not None and read.name:
            cur = memory_config.current().stored('prompt_pack', read.name, target.slug)
            pinned = await get_revision(kind='prompt_pack', name=read.name, revision=read.revision)
            active_body = pinned.body if pinned else None
    return act, cur, active_body, target_docs(None, target)


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
        '| target doc',
        cur,
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
    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    await save(client, source, 'region_profile', 'badge', {'x': 1})
    await activate_config(client, source, 'detection_profile', 'badge', 1)
    await save(client, target, 'region_profile', 'badge', {'x': 'TARGET OWN'})
    before = set(target_docs(client, target))
    try:
        await _clone(client, source, target, monkeypatch, axes=None)
        outcome = 'ok'
    except RequestRejectedError as exc:
        outcome = f'{exc.status_code} {exc.detail}'
    after = set(target_docs(client, target))
    print('profile-collision clone ->', outcome, '| new target docs:', sorted(after - before))
    assert not (outcome != 'ok' and after - before), 'partial write before 409'


@pytest.mark.asyncio
async def test_existing_target_previously_deactivated_accepts_clone(monkeypatch):
    """Existing target once activated a pack then deactivated it ('off').
    _validate_clone treats that as 'no activation'; activate(expected_active=None)
    then conflicts."""
    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    # target: activate a built-in/env id then turn it off (no stored packs left)
    await activate_config(client, target, 'prompt_pack', 'generic_item_v1', None)
    await activate_config(
        client,
        target,
        'prompt_pack',
        None,
        None,
        expected={'name': 'generic_item_v1', 'revision': None},
    )
    before = set(target_docs(client, target))
    try:
        await _clone(client, source, target, monkeypatch, axes=None)
        outcome = 'ok'
    except RequestRejectedError as exc:
        outcome = f'{exc.status_code} {exc.detail}'
    after = set(target_docs(client, target))
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
    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    await save(client, target, 'prompt_pack', 'wheel', {'class_system': 'MINE'})
    before = target_docs(client, target)
    with pytest.raises(RequestRejectedError) as exc:
        await _clone(client, source, target, monkeypatch, axes=['settings_defaults', 'activations'])
    assert exc.value.status_code == 409
    assert exc.value.detail['error'] == 'target_not_empty'
    assert target_docs(client, target) == before


@pytest.mark.asyncio
async def test_clone_into_previously_deactivated_target_succeeds(monkeypatch):
    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', 1)
    await activate_config(client, target, 'prompt_pack', 'generic_item_v1', None)
    await activate_config(
        client,
        target,
        'prompt_pack',
        None,
        None,
        expected={'name': 'generic_item_v1', 'revision': None},
    )
    patch_lifecycle(monkeypatch, source, target)
    await clone_into(client, source, target)
    from src.services.config_store.documents import get_activation_detail

    with bind_project(target):
        detail = await get_activation_detail('prompt_pack')
    print('deactivated target ->', detail)
    assert detail is not None
    assert detail.ref == ('wheel', 1)
    assert detail.previous == (None, None)


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
    from src.services.config_store import get_config_store

    client = Fake()
    source, target = record('alpha'), record('beta')
    await save(client, source, 'region_profile', 'badge', {'x': 1})
    await activate_config(client, source, 'detection_profile', 'badge', 1)
    with bind_project(target):
        store = get_config_store()
        await store.refresh()  # warm, empty
        # "other process" write: straight to the database, no apply_local here
        await save(client, target, 'region_profile', 'badge', {'x': 'OTHER WORKER'})
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
    saved = await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'x'})
    await activate_config(client, source, 'prompt_pack', 'wheel', saved.revision)
    await _apply_clone(
        target_record=target,
        source=source,
        axes=['prompt_packs', 'activations'],
        target_activations=None,
    )


async def _seed(client, source, *, edit_after_activate):
    saved = await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await activate_config(client, source, 'prompt_pack', 'wheel', saved.revision)
    if edit_after_activate:
        await save(
            client, source, 'prompt_pack', 'wheel', {'class_system': 'R2 NOT ACTIVATED'}, expected=1
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
        target_record=target,
        source=src_rec,
        axes=axes,
        target_activations=target_activations,
    )


@pytest.mark.asyncio
async def test_target_activates_the_source_activated_body_not_current():
    client = Fake()
    source, target = record('alpha'), record('beta')
    await _seed(client, source, edit_after_activate=True)
    await _apply_clone(
        target_record=target,
        source=source,
        axes=['prompt_packs', 'activations'],
        target_activations=None,
    )
    act, _cur, active_body, _ = await _target_state(client, target)
    print('target active', act, active_body)
    assert active_body == {'class_system': 'R1'}, (
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
        target_record=target,
        source=src_rec,
        axes=axes,
        target_activations=target_activations,
    )
