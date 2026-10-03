"""W4: ``GET /config/vocabulary`` (any_domain_plan.md §7.4)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.routers.curation.config_vocabulary import CHOICE_SOURCES


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
    from src.services.triton_control import TritonControlService

    async def _fake_repo_index() -> list[dict]:
        return [
            {'name': 'my_detector', 'state': 'READY', 'version': '1'},
            {'name': 'paddleocr_det_trt', 'state': 'READY', 'version': '1'},
            {'name': 'paddleocr_rec_trt', 'state': 'READY', 'version': '1'},
            {'name': 'ocr_pipeline', 'state': 'READY', 'version': '1'},
        ]

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr(
        TritonControlService, 'get_repository_index', lambda _self: _fake_repo_index()
    )

    fake_os = FakeConfigOpenSearch()
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def test_vocabulary_shape(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/config/vocabulary')
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {
        'detectors',
        'segmenters',
        'vlm',
        'ocr',
        'model_choices',
        'text_reader_modes',
        'registry_classes',
        'prompt_pack_calls',
        'labels',
        'reprocess',
    }


def test_vocabulary_ocr_lists_have_configured_first(app_client: TestClient) -> None:
    body = app_client.get('/curation/projects/default/config/vocabulary').json()
    assert body['ocr']['pipeline_models'][0]['configured'] is True
    assert body['ocr']['det_models'][0]['configured'] is True
    assert body['ocr']['rec_models'][0]['configured'] is True


def test_vocabulary_vlm_no_top_level_model_key(app_client: TestClient) -> None:
    body = app_client.get('/curation/projects/default/config/vocabulary').json()
    assert set(body['vlm']) == {'active', 'endpoints'}
    for entry in body['vlm']['endpoints']:
        assert set(entry) >= {
            'name',
            'source',
            'model',
            'resolved_model',
            'locality',
            'sends_images_externally',
            'status',
            'active',
        }


def test_vocabulary_vlm_absent_when_not_configured(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_VLM_URL', raising=False)
    body = app_client.get('/curation/projects/default/config/vocabulary').json()
    assert body['vlm']['active']['name'] is None
    assert body['vlm']['endpoints'] == []


def test_vocabulary_registry_classes_choice_id_is_name(app_client: TestClient) -> None:
    body = app_client.get('/curation/projects/default/config/vocabulary').json()
    for entry in body['registry_classes']:
        assert entry['choice']['id'] == entry['class_name']


def test_vocabulary_promoted_and_triton_merge(app_client: TestClient) -> None:
    body = app_client.get('/curation/projects/default/config/vocabulary').json()
    names = {d['name'] for d in body['detectors']}
    assert 'my_detector' in names
    entry = next(d for d in body['detectors'] if d['name'] == 'my_detector')
    assert entry['source'] == 'triton'
    assert entry['ready'] is True


def test_choice_sources_resolve_through_the_vocabulary_response(app_client: TestClient) -> None:
    """§7.3: every ``choices_from`` value maps (through CHOICE_SOURCES) to
    a list that actually exists on this response."""
    body = app_client.get('/curation/projects/default/config/vocabulary').json()
    for path in CHOICE_SOURCES.values():
        target: object = body
        for part in path.split('.'):
            assert isinstance(target, dict), f'{path} does not resolve on the vocabulary response'
            assert part in target, f'{path}: missing key {part!r}'
            target = target[part]
        assert isinstance(target, list)
        for entry in target:
            assert 'choice' in entry
            assert 'id' in entry['choice']
            assert 'label' in entry['choice']


def test_prompt_pack_calls_match_reply_key_contract(app_client: TestClient) -> None:
    from src.services.labeling.vlm_prompts import REPLY_KEY_CONTRACT

    body = app_client.get('/curation/projects/default/config/vocabulary').json()
    ids = {c['id'] for c in body['prompt_pack_calls']}
    assert ids == set(REPLY_KEY_CONTRACT)


def test_vocabulary_serves_the_reprocess_block(app_client: TestClient) -> None:
    body = app_client.get('/curation/projects/default/config/vocabulary').json()['reprocess']
    assert {e['id'] for e in body['scopes']} >= {'detect', 'region', 'embed'}
    assert {e['id'] for e in body['lock_reasons']} >= {'human_label', 'validated', 'imported'}
    assert all(e['label'] for e in body['job_statuses'])
