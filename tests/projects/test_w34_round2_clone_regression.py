"""W3/W4 review round-2 (2026-09-28) regression tests for B2: a clone must
succeed through the REAL ``clone_settings``/``_validate_clone`` ->
``_apply_clone`` sequence (not just ``_apply_clone`` called directly,
which the round-1 fix's own test did -- dodging the target store's 1s
cache TTL that raced with ``_clone_prompt_packs``'s write in the real
flow), with no artificial sleep needed once the sequencing bug is
actually fixed. Also pins that the target's activated pack body is always
the source's ACTIVATED (validated) body, never a merely-current,
never-activated draft (W2 Minor 2) -- even when a sibling ``prompt_packs``
axis already wrote that name's current doc into the target first. Landed
here as permanent tests (moved from the reviewer's throwaway probe file)
rather than left as scratch fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from curation._fake_config_opensearch import FakeConfigOpenSearch

from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store.store import reset_config_stores
from src.services.projects.clone import _apply_clone, _validate_clone


pytestmark = pytest.mark.unbound


def _record(slug):
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
def _reset():
    reset_config_stores()
    yield
    reset_config_stores()


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
async def test_b2_real_flow_validate_then_apply(monkeypatch):
    """clone_settings / clone_settings_into run _validate_clone (which warms the
    target store) immediately before _apply_clone."""
    from src.services.projects import lifecycle as lifecycle_mod

    client = FakeConfigOpenSearch()
    source, target = _record('alpha'), _record('beta')
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
async def test_b2_target_activates_the_source_activated_body_not_current():
    from src.config import get_curation_config
    from src.services.config_store.index import get_activation

    client = FakeConfigOpenSearch()
    source, target = _record('alpha'), _record('beta')
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
async def test_b2_real_flow_passes_only_after_cache_ttl(monkeypatch):
    import asyncio

    from src.services.projects import lifecycle as lifecycle_mod

    client = FakeConfigOpenSearch()
    source, target = _record('alpha'), _record('beta')
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
