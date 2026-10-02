"""W2: ``PUT/GET /curation/settings`` bridges ``prompt_pack`` /
``detection_profile`` through the config store's activation instead of
the generic settings-doc ``defaults`` map --
any_domain_plan.md §3.7/§9 W2."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    from src.clients import curation_opensearch
    from src.services.config_store.store import reset_config_stores
    from src.services.curation.strategy_registry import _reset_field_coverage_cache

    curation_opensearch._settings_cache.clear()
    reset_config_stores()
    _reset_field_coverage_cache()
    yield
    curation_opensearch._settings_cache.clear()
    reset_config_stores()
    _reset_field_coverage_cache()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeConfigOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def test_put_prompt_pack_activates_through_store(app_client: TestClient) -> None:
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    r = app_client.put(
        '/curation/projects/default/settings',
        json={'defaults': {'prompt_pack': GENERIC_ITEM_PACK.name}},
    )
    assert r.status_code == 200, r.text
    assert r.json()['defaults']['prompt_pack'] == GENERIC_ITEM_PACK.name

    r2 = app_client.get('/curation/projects/default/settings')
    assert r2.json()['defaults']['prompt_pack'] == GENERIC_ITEM_PACK.name


def test_put_prompt_pack_unknown_id_422(app_client: TestClient) -> None:
    r = app_client.put(
        '/curation/projects/default/settings', json={'defaults': {'prompt_pack': 'not_a_pack'}}
    )
    assert r.status_code == 422
    detail = r.json()['detail']
    assert detail['error'] == 'unknown_pack'
    assert detail['axis'] == 'prompt_pack'


def test_put_prompt_pack_null_deactivates(app_client: TestClient) -> None:
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    app_client.put(
        '/curation/projects/default/settings',
        json={'defaults': {'prompt_pack': GENERIC_ITEM_PACK.name}},
    )
    r = app_client.put(
        '/curation/projects/default/settings', json={'defaults': {'prompt_pack': None}}
    )
    assert r.status_code == 200, r.text
    # Minor 4 (W2 review): `null` deactivates through the store exactly
    # like `'off'` does (`_activate_config_store_axis` maps both to
    # `name=None`) -- GET must report that real 'off' state, not omit it
    # as if the axis had never been touched.
    assert r.json()['defaults']['prompt_pack'] == 'off'


def test_put_detection_profile_off_and_on(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W2: detection_profile is now settable through the store, unlike
    the prior read-only behavior."""
    from src.config import DetectionProfile
    from src.services.detection import profile_registry

    # N1 fix (W3/W4 round-3 review): `PUT /settings` now runs the same
    # `for_activation` validation `POST /{name}/activate` runs -- a bare
    # `DetectionProfile(name='wheel')` (no detector, no segmenter) can
    # never produce a box and genuinely fails that gate now, same as it
    # would on the direct activate route. `text_reader='none'` skips the
    # OCR-model checks (this test is about the settings-bridge plumbing,
    # not OCR); the detector model is reported READY the same way
    # `test_region_profiles_router.py`'s `app_client` fixture does.
    async def _fake_repo_index() -> list[dict]:
        return [{'name': 'wheel_detector', 'state': 'READY', 'version': '1'}]

    from src.services.triton_control import TritonControlService

    monkeypatch.setattr(
        TritonControlService, 'get_repository_index', lambda _self: _fake_repo_index()
    )

    profile_registry._reset_registry_for_tests()
    profile_registry.register_profile(
        DetectionProfile(name='wheel', detector_model='wheel_detector', text_reader='none'),
        default=True,
    )
    try:
        r = app_client.put(
            '/curation/projects/default/settings', json={'defaults': {'detection_profile': 'wheel'}}
        )
        assert r.status_code == 200, r.text
        assert r.json()['defaults']['detection_profile'] == 'wheel'

        r_off = app_client.put(
            '/curation/projects/default/settings', json={'defaults': {'detection_profile': 'off'}}
        )
        assert r_off.status_code == 200, r_off.text
        # Minor 4 (W2 review): an explicit deactivation is reported as
        # 'off', not omitted -- the store's own docstring says those two
        # states (never activated vs. explicitly turned off) are
        # deliberately distinct, and GET must not collapse them.
        assert r_off.json()['defaults']['detection_profile'] == 'off'

        # 'off' is a true deactivation -- the env-registered default does
        # NOT silently take back over.
        from src.services.detection.profile_registry import get_active_region_profile

        assert get_active_region_profile() is None
    finally:
        profile_registry._reset_registry_for_tests()


def test_other_axes_still_use_the_generic_settings_doc(app_client: TestClient) -> None:
    r = app_client.put('/curation/projects/default/settings', json={'defaults': {'cluster': 'ahc'}})
    assert r.status_code == 200
    assert r.json()['defaults']['cluster'] == 'ahc'


def test_active_conflict_is_structured_409(app_client: TestClient) -> None:
    """A second activation racing the same axis without re-reading the
    current activation gets a structured 409, not a stack trace."""
    from src.services.config_store.store import get_config_store
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, GENERIC_REGION_PACK

    app_client.put(
        '/curation/projects/default/settings',
        json={'defaults': {'prompt_pack': GENERIC_ITEM_PACK.name}},
    )
    # Force the process-local snapshot to disagree with what's actually
    # stored (simulating a second writer's activation this process
    # hasn't seen yet) while keeping it "fresh enough" (a real
    # ``loaded_at``) that ``ensure_fresh`` serves it from cache instead
    # of re-fetching and silently repairing it before the route reads it.
    import time as _time

    store = get_config_store()
    store.current = store.current.__class__(
        config_revision=store.current.config_revision,
        active_pack=None,
        loaded_at=_time.monotonic(),
    )
    r = app_client.put(
        '/curation/projects/default/settings',
        json={'defaults': {'prompt_pack': GENERIC_REGION_PACK.name}},
    )
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'active_conflict'


def test_put_prompt_pack_activates_a_stored_pack_at_its_real_revision(
    app_client: TestClient,
) -> None:
    """M6: activating a STORED pack (not an env/file id) through the
    settings bridge must stamp its own current revision, not `None` --
    every other process's `_axis_ref` read used to coerce a `None`
    revision to `0`, so two processes disagreed about which revision
    was active."""
    from src.config.project_context import bind_project
    from src.services.config_store.index import save_config
    from src.services.config_store.store import ConfigStore, reset_config_stores
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    fake_os = app_client.fake_os  # type: ignore[attr-defined]

    async def _seed() -> tuple[int, str]:
        from src.config.curation import IndexRole, base_curation_config
        from src.config.projects import ProjectRecord, resources_for_new

        record = ProjectRecord(
            slug='default',
            display_name='Default',
            description='',
            status='active',
            revision=1,
            created_at='',
            updated_at='',
            origin=None,
            resources=resources_for_new('default', base_curation_config()),
        )
        with bind_project(record):
            idx = resources_for_new('default', base_curation_config()).indexes[IndexRole.CONFIGS]
            body = {**GENERIC_ITEM_PACK.to_dict(), 'name': 'stored_pack'}
            doc = await save_config(
                fake_os,
                idx,
                kind='prompt_pack',
                name='stored_pack',
                body=body,
                expected_revision=None,
            )
            return doc['revision'], idx

    import asyncio

    revision, idx = asyncio.run(_seed())

    r = app_client.put(
        '/curation/projects/default/settings',
        json={'defaults': {'prompt_pack': 'stored_pack'}},
    )
    assert r.status_code == 200, r.text

    # A fresh ConfigStore (simulating another process that never wrote
    # this activation itself) reads the activation doc from scratch.
    reset_config_stores()
    other_process_store = ConfigStore(index=idx, mode='live')

    async def _refresh() -> None:
        await other_process_store.refresh(fake_os)

    asyncio.run(_refresh())
    assert other_process_store.current.active_pack == ('stored_pack', revision)
    assert other_process_store.current.active_pack != ('stored_pack', 0)


@pytest.mark.asyncio
async def test_settings_doc_key_cannot_override_the_active_pack() -> None:
    """A stale ``defaults.prompt_pack`` in the settings document (written
    before the axis moved to the config store) must not change which pack
    an omitted-``prompt_pack`` run resolves to."""
    from src.services.curation.strategy_defaults import resolve_effective_default
    from src.services.labeling.vlm_prompts import GENERIC_REGION_PACK, active_prompt_pack

    assert active_prompt_pack().name != GENERIC_REGION_PACK.name
    stale = {'defaults': {'prompt_pack': GENERIC_REGION_PACK.name}}
    resolved = await resolve_effective_default('prompt_pack', settings_doc=stale)
    assert resolved == active_prompt_pack().name


def test_get_settings_hides_a_stale_config_store_key(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = AsyncMock(
        return_value={'defaults': {'prompt_pack': 'ghost'}, 'updated_at': None, 'updated_by': None}
    )
    monkeypatch.setattr('src.clients.curation_opensearch.get_curation_settings', stale)
    r = app_client.get('/curation/projects/default/settings')
    assert 'ghost' not in r.json()['defaults'].values()
