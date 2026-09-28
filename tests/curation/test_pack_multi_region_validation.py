"""W3: the multi-box pairing check between a flat (N=1-shaped) pack and
an N>1 active region profile (any_domain_plan.md §3.3/§4.3)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.services.config_store.pack_validation import validate_pack
from src.services.labeling.vlm_prompts import GENERIC_REGION_PACK


def _flat_pack_body() -> dict[str, object]:
    """A pre-W8-shaped pack: the combined calls name the per-box verdict
    keys as top-level (flat) fields, never ``region_boxes``/``box`` --
    valid for N=1, missing the W8 multi-box list keys."""
    body = GENERIC_REGION_PACK.to_dict()
    body.pop('name')
    body['combined_system'] = (
        'Return STRICT JSON: class_id, class_confidence, region_visible, '
        'region_bbox_correct, region_confidence.'
    )
    body['combined_user_template'] = '{class_block}{region_block}Answer with the fields above.'
    body['combined_batch_system'] = (
        'Return STRICT JSON: {"results": [{"img": 1, "class_id": 0, "class_confidence": "high", '
        '"region_visible": true, "region_bbox_correct": true, "region_confidence": "high"}]}'
    )
    body['combined_batch_rules'] = 'Same fields per image, in a "results" array.'
    return body


def test_flat_pack_against_n4_profile_warns_on_validate() -> None:
    from src.config import DetectionProfile

    profile = DetectionProfile(name='p', max_regions_per_item=4)
    report = validate_pack(None, _flat_pack_body(), profile=profile, for_activation=False)
    assert report.ok  # warning, not error
    codes = {w.code for w in report.warnings}
    assert 'pack_multi_region_keys_missing' in codes


def test_flat_pack_against_n4_profile_errors_on_activation_and_force_does_not_bypass() -> None:
    from src.config import DetectionProfile

    profile = DetectionProfile(name='p', max_regions_per_item=4)
    report = validate_pack(None, _flat_pack_body(), profile=profile, for_activation=True)
    assert not report.ok
    assert any(e.code == 'pack_multi_region_keys_missing' for e in report.errors)
    assert all(
        not e.bypassable for e in report.errors if e.code == 'pack_multi_region_keys_missing'
    )
    assert report.force_allowed is False


def test_flat_pack_against_n1_profile_has_no_issue() -> None:
    from src.config import DetectionProfile

    profile = DetectionProfile(name='p', max_regions_per_item=1)
    report = validate_pack(None, _flat_pack_body(), profile=profile, for_activation=True)
    codes = {e.code for e in report.errors} | {w.code for w in report.warnings}
    assert 'pack_multi_region_keys_missing' not in codes


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores
    from src.services.detection import profile_registry

    reset_config_stores()
    profile_registry._reset_registry_for_tests()
    yield
    reset_config_stores()
    profile_registry._reset_registry_for_tests()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeConfigOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def test_validate_route_profile_query_param_n4_warns(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import DetectionProfile
    from src.services.detection import profile_registry

    profile_registry.register_profile(
        DetectionProfile(name='n4profile', max_regions_per_item=4), default=True
    )
    r = app_client.post(
        '/curation/projects/default/prompt_packs/validate?profile=n4profile',
        json={'name': None, 'body': _flat_pack_body()},
    )
    assert r.status_code == 200, r.text
    codes = {w['code'] for w in r.json()['warnings']}
    assert 'pack_multi_region_keys_missing' in codes
