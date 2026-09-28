"""Round-3 reviewer probes, landed as permanent regression tests (round 4
confirmed all fixed): REAL clone_settings flow (validate -> apply), no sleep."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from curation._fake_config_opensearch import FakeConfigOpenSearch
from fastapi import HTTPException

from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store.store import reset_config_stores


pytestmark = pytest.mark.unbound


class Fake(FakeConfigOpenSearch):
    async def count(self, index: str, body: Any = None) -> dict[str, Any]:  # noqa: ARG002
        return {'count': 0}

    async def bulk(self, body: list[Any], refresh: Any = False) -> dict[str, Any]:  # noqa: ARG002
        return {'errors': False, 'items': []}


def _record(slug, tmp_path):
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


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation as curation_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    reset_config_stores()
    yield
    reset_config_stores()
    curation_mod._default_curation_config = None


async def _save(client, rec, kind, name, body, expected=None):
    from src.config import get_curation_config
    from src.services.config_store.index import save_config

    with bind_project(rec):
        return await save_config(
            client,
            get_curation_config().configs_index,
            kind=kind,
            name=name,
            body=body,
            expected_revision=expected,
        )


async def _activate(client, rec, axis, name, revision, expected=None):
    from src.config import get_curation_config
    from src.services.config_store.index import activate

    with bind_project(rec):
        return await activate(
            client,
            get_curation_config().configs_index,
            axis=axis,
            name=name,
            revision=revision,
            expected_active=expected,
        )


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
async def test_r3_real_clone_settings_no_409(tmp_path, monkeypatch, axes):
    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    await _save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await _activate(client, source, 'prompt_pack', 'wheel', 1)
    await _clone(client, source, target, monkeypatch, axes=axes)
    act, _cur, active_body, _ = await _target_state(client, target)
    print('target act', act and (act['name'], act['revision']), 'body', active_body)
    assert active_body == {'class_system': 'R1'}


@pytest.mark.asyncio
async def test_r3_real_clone_draft_not_promoted(tmp_path, monkeypatch):
    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    await _save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await _activate(client, source, 'prompt_pack', 'wheel', 1)
    await _save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R2 DRAFT'}, expected=1)
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
async def test_r3_draft_only_source_pack_never_active_in_target(tmp_path, monkeypatch):
    """Source has a pack that was PUT but never activated at all."""
    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    await _save(client, source, 'prompt_pack', 'never_active', {'class_system': 'D'})
    await _clone(client, source, target, monkeypatch, axes=None)
    act, *_ = await _target_state(client, target)
    print('draft-only source -> target activation', act)
    assert not (act and act.get('name'))


# ---- existing-target (clone_settings_into-shaped) residual gaps ---------------
@pytest.mark.asyncio
async def test_r3_existing_target_with_same_named_unactivated_profile(tmp_path, monkeypatch):
    """Existing target has an un-activated stored region profile with the same
    name as the source's ACTIVE profile. Expect: clean 409 BEFORE any write."""
    from src.config import get_curation_config

    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    await _save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await _activate(client, source, 'prompt_pack', 'wheel', 1)
    await _save(client, source, 'region_profile', 'badge', {'x': 1})
    await _activate(client, source, 'detection_profile', 'badge', 1)
    await _save(client, target, 'region_profile', 'badge', {'x': 'TARGET OWN'})
    with bind_project(target):
        tidx = get_curation_config().configs_index
    before = set(client._docs.get(tidx, {}))
    try:
        await _clone(client, source, target, monkeypatch, axes=None)
        outcome = 'ok'
    except HTTPException as exc:
        outcome = f'{exc.status_code} {exc.detail}'
    after = set(client._docs.get(tidx, {}))
    print('profile-collision clone ->', outcome, '| new target docs:', sorted(after - before))
    assert not (outcome != 'ok' and after - before), 'partial write before 409'


@pytest.mark.asyncio
async def test_r3_existing_target_previously_deactivated(tmp_path, monkeypatch):
    """Existing target once activated a pack then deactivated it ('off').
    _validate_clone treats that as 'no activation'; activate(expected_active=None)
    then conflicts."""
    from src.config import get_curation_config

    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    await _save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await _activate(client, source, 'prompt_pack', 'wheel', 1)
    # target: activate a built-in/env id then turn it off (no stored packs left)
    await _activate(client, target, 'prompt_pack', 'generic_item', None)
    await _activate(
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
    except HTTPException as exc:
        outcome = f'{exc.status_code} {exc.detail}'
    after = set(client._docs.get(tidx, {}))
    print('deactivated-target clone ->', outcome, '| new target docs:', sorted(after - before))
    # m5 fix (W3/W4 round-4 review): a target that once deactivated this
    # axis is genuinely EMPTY (no stored pack of its own on this axis) --
    # a clone into it is not just "must not partially write", it must
    # actually SUCCEED. The old `outcome == 'ok' or not (after - before)`
    # let a clean 409 (no docs written but also nothing cloned) pass
    # silently, masking the real N3a bug (a spurious `active_conflict`
    # from treating 'off' as `expected_active=None`) instead of proving
    # it is fixed.
    assert outcome == 'ok', f'clone into a previously-deactivated target must succeed: {outcome}'
