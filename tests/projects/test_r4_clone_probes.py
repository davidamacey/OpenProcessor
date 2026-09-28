"""Round-4 reviewer probes: N3 through the REAL clone_settings_into entry point."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from projects.test_r3_clone_probes import (  # noqa: F401 - reuse fixtures/helpers
    Fake,
    _activate,
    _env,
    _record,
    _save,
)
from src.config.project_context import bind_project


pytestmark = pytest.mark.unbound


def _patch_lifecycle(monkeypatch, source, target):
    from src.services.projects import lifecycle as lifecycle_mod, registry as registry_mod

    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=source))
    monkeypatch.setattr(
        lifecycle_mod, '_get_mutable_record', AsyncMock(return_value=(target, 1, 1))
    )
    monkeypatch.setattr(lifecycle_mod, 'write_record', AsyncMock(return_value=None))
    reg = MagicMock()
    reg.ensure_fresh = AsyncMock(return_value=None)
    monkeypatch.setattr(registry_mod, 'get_project_registry', lambda: reg)


def _docs(client, rec):
    from src.config import get_curation_config

    with bind_project(rec):
        idx = get_curation_config().configs_index
    return dict(client._docs.get(idx, {}))


async def _into(client, source, target):
    from src.services.projects.clone import clone_settings_into

    return await clone_settings_into(
        client, slug=target.slug, from_slug=source.slug, axes=None, expected_revision=1
    )


@pytest.mark.asyncio
async def test_r4_into_previously_deactivated_target_succeeds(tmp_path, monkeypatch):
    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    await _save(client, source, 'prompt_pack', 'wheel', {'class_system': 'R1'})
    await _activate(client, source, 'prompt_pack', 'wheel', 1)
    await _activate(client, target, 'prompt_pack', 'generic_item', None)
    await _activate(
        client,
        target,
        'prompt_pack',
        None,
        None,
        expected={'name': 'generic_item', 'revision': None},
    )
    _patch_lifecycle(monkeypatch, source, target)
    await _into(client, source, target)
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
async def test_r4_into_same_named_unactivated_config_409_before_writes(
    tmp_path, monkeypatch, kind, axis, name
):
    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    await _save(client, source, kind, name, {'x': 1})
    await _activate(client, source, axis, name, 1)
    await _save(client, target, kind, name, {'x': 'TARGET OWN'})
    _patch_lifecycle(monkeypatch, source, target)
    before = _docs(client, target)
    with pytest.raises(HTTPException) as ei:
        await _into(client, source, target)
    after = _docs(client, target)
    changed = sorted(k for k in set(after) | set(before) if after.get(k) != before.get(k))
    print(kind, '->', ei.value.status_code, ei.value.detail, '| changed:', changed)
    assert ei.value.status_code == 409
    assert changed == []


@pytest.mark.asyncio
async def test_r4_into_stale_target_cache_race(tmp_path, monkeypatch):
    """Another worker process stored 'badge' in the target <1s after this
    process last warmed the target store. Validation reads the warm cache."""
    from src.config import get_curation_config
    from src.services.config_store import get_config_store
    from src.services.config_store.index import save_config

    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    await _save(client, source, 'region_profile', 'badge', {'x': 1})
    await _activate(client, source, 'detection_profile', 'badge', 1)
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
    _patch_lifecycle(monkeypatch, source, target)
    before = _docs(client, target)
    try:
        await _into(client, source, target)
        outcome = 'ok'
    except HTTPException as exc:
        outcome = f'{exc.status_code} {exc.detail}'
    after = _docs(client, target)
    changed = sorted(k for k in set(after) | set(before) if after.get(k) != before.get(k))
    print('stale-cache race ->', outcome, '| changed:', changed)
    assert outcome == 'ok' or changed == [], 'partial write before 409'
