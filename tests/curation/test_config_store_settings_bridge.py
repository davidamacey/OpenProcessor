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
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def test_put_prompt_pack_activates_through_store(app_client: TestClient) -> None:
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    r = app_client.put(
        '/curation/settings', json={'defaults': {'prompt_pack': GENERIC_ITEM_PACK.name}}
    )
    assert r.status_code == 200, r.text
    assert r.json()['defaults']['prompt_pack'] == GENERIC_ITEM_PACK.name

    r2 = app_client.get('/curation/settings')
    assert r2.json()['defaults']['prompt_pack'] == GENERIC_ITEM_PACK.name


def test_put_prompt_pack_unknown_id_422(app_client: TestClient) -> None:
    r = app_client.put('/curation/settings', json={'defaults': {'prompt_pack': 'not_a_pack'}})
    assert r.status_code == 422
    detail = r.json()['detail']
    assert detail['error'] == 'unknown_pack'
    assert detail['axis'] == 'prompt_pack'


def test_put_prompt_pack_null_deactivates(app_client: TestClient) -> None:
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    app_client.put('/curation/settings', json={'defaults': {'prompt_pack': GENERIC_ITEM_PACK.name}})
    r = app_client.put('/curation/settings', json={'defaults': {'prompt_pack': None}})
    assert r.status_code == 200, r.text
    assert 'prompt_pack' not in r.json()['defaults']


def test_put_detection_profile_off_and_on(app_client: TestClient) -> None:
    """W2: detection_profile is now settable through the store, unlike
    the prior read-only behavior."""
    from src.config import DetectionProfile
    from src.services.detection import profile_registry

    profile_registry._reset_registry_for_tests()
    profile_registry.register_profile(DetectionProfile(name='wheel'), default=True)
    try:
        r = app_client.put('/curation/settings', json={'defaults': {'detection_profile': 'wheel'}})
        assert r.status_code == 200, r.text
        assert r.json()['defaults']['detection_profile'] == 'wheel'

        r_off = app_client.put(
            '/curation/settings', json={'defaults': {'detection_profile': 'off'}}
        )
        assert r_off.status_code == 200, r_off.text
        assert 'detection_profile' not in r_off.json()['defaults']

        # 'off' is a true deactivation -- the env-registered default does
        # NOT silently take back over.
        from src.services.detection.profile_registry import get_active_region_profile

        assert get_active_region_profile() is None
    finally:
        profile_registry._reset_registry_for_tests()


def test_other_axes_still_use_the_generic_settings_doc(app_client: TestClient) -> None:
    r = app_client.put('/curation/settings', json={'defaults': {'cluster': 'ahc'}})
    assert r.status_code == 200
    assert r.json()['defaults']['cluster'] == 'ahc'


def test_active_conflict_is_structured_409(app_client: TestClient) -> None:
    """A second activation racing the same axis without re-reading the
    current activation gets a structured 409, not a stack trace."""
    from src.services.config_store.store import get_config_store
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, GENERIC_REGION_PACK

    app_client.put('/curation/settings', json={'defaults': {'prompt_pack': GENERIC_ITEM_PACK.name}})
    # Force the process-local snapshot stale so the route re-derives an
    # ``expected_active`` that no longer matches what's actually stored.
    store = get_config_store()
    store.current = store.current.__class__(
        config_revision=store.current.config_revision, active_pack=None
    )
    r = app_client.put(
        '/curation/settings', json={'defaults': {'prompt_pack': GENERIC_REGION_PACK.name}}
    )
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'active_conflict'
