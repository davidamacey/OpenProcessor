"""``GET /regions/vocabulary`` and ``GET /review/tabs`` (W0 of
``docs/design/naming_sweep_plan.md`` -- finding m9).

After S3/S7 the detector/segmenter/VLM identifiers the worker writes come
entirely from deployment config, so the frontend can no longer hardcode a
label/palette map keyed on ``lpr_nanov11_640`` / ``sam3`` / ``gemma-4-e4b``.
These two endpoints are the served vocabulary a client renders from
instead.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from _region_profile_fixture import EXAMPLE_LICENSE_PLATE_PROFILE_PATH
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config.region_source import CANDIDATE_SOURCES
from src.services.curation.review_queries import KNOWN_TABS


@pytest.fixture
def client() -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    # C3: GET /review/tabs now also serves empty_state, which issues a
    # couple of `count` calls against opensearch.
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 0})
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def test_regions_vocabulary_has_no_active_profile_by_default(client: TestClient) -> None:
    """No-profile gating contract: with no active region profile, the
    endpoint still 200s (never 404s) and serves `region_profile: null`
    plus every list empty -- not a degraded-but-populated catalog."""
    resp = client.get('/curation/regions/vocabulary')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {
        'region_profile',
        'detectors',
        'region_sources',
        'chain_actors',
        'text_rules',
        'text_choices',
        'rejection_reasons',
    }
    assert body['region_profile'] is None
    assert body['text_rules'] is None
    assert body['detectors'] == []
    assert body['region_sources'] == []
    assert body['chain_actors'] == []
    assert body['text_choices'] == []
    assert body['rejection_reasons'] == []


def test_regions_vocabulary_reflects_the_active_profile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.detection import profile_registry

    monkeypatch.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    profile_registry._reset_registry_for_tests()
    try:
        resp = client.get('/curation/regions/vocabulary')
        assert resp.status_code == 200, resp.text
        body = resp.json()
        by_id = {d['id']: d for d in body['detectors']}
        assert by_id['license_plate_detector']['role'] == 'detector'
        assert by_id['license_plate_detector']['filterable'] is True
        assert by_id['sam3']['role'] == 'segmenter'
        assert by_id['sam3']['filterable'] is True
        # OCR text-hint locates text but never sets the region bbox --
        # never filterable.
        assert by_id['paddleocr_det_trt']['role'] == 'ocr'
        assert by_id['paddleocr_det_trt']['filterable'] is False
        # VLM model comes from OP_VLM_MODEL (conftest.py sets it globally
        # for the suite), never a hardcoded default.
        assert by_id['test-vlm-model']['role'] == 'verifier'
        assert by_id['test-vlm-model']['filterable'] is False
    finally:
        profile_registry._reset_registry_for_tests()


def test_regions_vocabulary_env_configured_detector_reflected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vocabulary reflects env-configured detector/VLM/segmenter names --
    swap the profile's detector_model via env and confirm the served
    vocabulary follows, proving nothing is hardcoded."""
    from src.services.detection import profile_registry

    monkeypatch.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    monkeypatch.setenv('OP_REGION_DETECTION_DETECTOR_MODEL', 'my_custom_region_yolo')
    profile_registry._reset_registry_for_tests()
    try:
        resp = client.get('/curation/regions/vocabulary')
        assert resp.status_code == 200, resp.text
        body = resp.json()
        detector_ids = {d['id'] for d in body['detectors']}
        assert 'my_custom_region_yolo' in detector_ids
        assert 'license_plate_detector' not in detector_ids
    finally:
        profile_registry._reset_registry_for_tests()


@pytest.mark.usefixtures('reference_region_profile')
def test_regions_vocabulary_covers_every_s3_region_source_value(client: TestClient) -> None:
    resp = client.get('/curation/regions/vocabulary')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    served_ids = {s['id'] for s in body['region_sources']}
    assert set(CANDIDATE_SOURCES).issubset(served_ids)
    assert 'human' in served_ids


def test_review_tabs_has_a_label_for_every_known_tab(client: TestClient) -> None:
    """No region profile is active in this fixture -- 'regions' is
    omitted entirely (no-profile gating contract)."""
    resp = client.get('/curation/review/tabs')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    served_ids = {t['id'] for t in body['tabs']}
    assert served_ids == set(KNOWN_TABS) - {'regions'}
    for tab in body['tabs']:
        assert tab['label']
        assert tab['description']
    by_id = {t['id']: t for t in body['tabs']}
    assert by_id['classifier_blind_spots']['label'] == 'Classifier blind spots'


def test_review_tabs_omits_regions_tab_without_a_profile(client: TestClient) -> None:
    resp = client.get('/curation/review/tabs')
    assert resp.status_code == 200, resp.text
    served_ids = {t['id'] for t in resp.json()['tabs']}
    assert 'regions' not in served_ids


@pytest.mark.usefixtures('reference_region_profile')
def test_review_tabs_includes_regions_tab_with_a_profile(client: TestClient) -> None:
    resp = client.get('/curation/review/tabs')
    assert resp.status_code == 200, resp.text
    served_ids = {t['id'] for t in resp.json()['tabs']}
    assert served_ids == set(KNOWN_TABS)


def test_review_tabs_regions_label_uses_the_active_profiles_display_name(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The active region profile's ``display_name`` (e.g. 'Plates') becomes
    the 'regions' tab's label, not the generic fallback."""
    from src.services.detection import profile_registry

    monkeypatch.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    profile_registry._reset_registry_for_tests()
    try:
        resp = client.get('/curation/review/tabs')
        assert resp.status_code == 200, resp.text
        by_id = {t['id']: t for t in resp.json()['tabs']}
        assert by_id['regions']['label'] == 'Plates'
    finally:
        profile_registry._reset_registry_for_tests()


@pytest.mark.usefixtures('reference_region_profile')
def test_review_tabs_serves_region_status_filter_spec(client: TestClient) -> None:
    """DQ-B2 follow-up: the regions tab's ``region_status`` filter is served
    as a self-describing enum spec (param, kind, label, value/label options)
    so the frontend renders any enum filter generically, with no per-filter
    code; its default rides the existing ``filter_defaults`` map."""
    resp = client.get('/curation/review/tabs')
    assert resp.status_code == 200, resp.text
    by_id = {t['id']: t for t in resp.json()['tabs']}
    regions = by_id['regions']
    assert 'region_status' in regions['filters']
    assert regions['filter_defaults']['region_status'] == 'all'
    specs = {s['param']: s for s in regions['filter_specs']}
    spec = specs['region_status']
    assert spec['kind'] == 'enum'
    assert spec['label']
    assert {o['value'] for o in spec['options']} == {'all', 'detected', 'verify_rejected'}
    assert all(o['label'] for o in spec['options'])
    assert regions['filter_defaults']['region_status'] in {o['value'] for o in spec['options']}
    for tab_id, tab in by_id.items():
        # Every spec'd filter is one the tab honours.
        assert {s['param'] for s in tab['filter_specs']} <= set(tab['filters'])
        if tab_id != 'regions':
            assert 'region_status' not in {s['param'] for s in tab['filter_specs']}


def test_review_tabs_response_is_typed_in_the_contract(client: TestClient) -> None:
    """The tab catalog (incl. ``filter_specs``) is a declared response model,
    so it lands in the generated OpenAPI contract the frontend vendors."""
    schema = client.get('/openapi.json').json()
    op = schema['paths']['/curation/review/tabs']['get']
    ref = op['responses']['200']['content']['application/json']['schema']
    assert '$ref' in ref, ref


def test_review_tabs_route_not_shadowed_by_the_tab_path_param(client: TestClient) -> None:
    """'/review/tabs' must resolve to the tab-catalog route, not
    'GET /review/{tab}' with tab='tabs' (an unknown tab -> 400)."""
    resp = client.get('/curation/review/tabs')
    assert resp.status_code == 200
    assert 'tabs' in resp.json()


@pytest.mark.usefixtures('reference_region_profile')
def test_regions_vocabulary_serves_the_region_text_rules(client: TestClient) -> None:
    rules = client.get('/curation/regions/vocabulary').json()['text_rules']
    assert rules['charset'] == '[A-Z0-9]'
    assert (rules['len_min'], rules['len_max']) == (2, 10)
    assert rules['reject_sequences'] is True
    assert 'NOTREADABLE' in rules['no_reading_words']
    assert 'placeholder' in rules['invalid_reasons']
    assert isinstance(rules['placeholders'], list)


@pytest.mark.usefixtures('reference_region_profile')
def test_regions_vocabulary_serves_the_rejection_reasons(client: TestClient) -> None:
    from src.config.region_rejection import (
        REJECT_REASON_NO_VERDICT,
        REJECT_REASON_SANITY_PREFIX,
        REJECT_REASON_VERIFIER,
    )

    reasons = client.get('/curation/regions/vocabulary').json()['rejection_reasons']
    by_id = {r['id']: r for r in reasons}
    assert set(by_id) == {
        REJECT_REASON_VERIFIER,
        REJECT_REASON_SANITY_PREFIX,
        REJECT_REASON_NO_VERDICT,
    }
    assert by_id[REJECT_REASON_VERIFIER] == {
        'id': 'region_visible_elsewhere',
        'label': 'Verifier: the box is wrong (region is elsewhere)',
        'kind': 'model_verdict',
        'match': 'exact',
        'label_template': None,
    }
    assert by_id[REJECT_REASON_SANITY_PREFIX]['match'] == 'prefix'
    assert by_id[REJECT_REASON_SANITY_PREFIX]['kind'] == 'automatic'
    assert by_id[REJECT_REASON_NO_VERDICT]['kind'] == 'needs_human'


def test_regions_vocabulary_response_is_typed_in_openapi(client: TestClient) -> None:
    """The frontend generates types from the OpenAPI contract: the route
    declares its response shape, including the rejection-reason kinds."""
    from src.config.region_rejection import REJECTION_REASON_KINDS

    spec = client.get('/openapi.json').json()
    ok = spec['paths']['/curation/regions/vocabulary']['get']['responses']['200']
    ref = ok['content']['application/json']['schema']['$ref'].rsplit('/', 1)[-1]
    schemas = spec['components']['schemas']
    props = schemas[ref]['properties']
    assert set(props) == {
        'region_profile',
        'detectors',
        'region_sources',
        'chain_actors',
        'text_rules',
        'text_choices',
        'rejection_reasons',
    }
    entry = schemas[props['rejection_reasons']['items']['$ref'].rsplit('/', 1)[-1]]
    assert set(entry['properties']) == {'id', 'label', 'kind', 'match', 'label_template'}
    assert set(entry['properties']['kind']['enum']) == set(REJECTION_REASON_KINDS)
    assert set(entry['properties']['match']['enum']) == {'exact', 'prefix'}
