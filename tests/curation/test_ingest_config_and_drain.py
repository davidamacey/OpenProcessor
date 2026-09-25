"""`GET /curation/ingest/config`. Also covers the typed drain verdict on
`GET /curation/ingest/region_drain`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation import region_drain


@pytest.fixture(autouse=True)
def _reset_drain():
    region_drain._reset_for_tests()
    yield
    region_drain._reset_for_tests()


def _client(fake: AsyncMock) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_ingest_config_is_typed_and_reflects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_UPLOAD_MAX_IMAGES_PER_REQUEST', '7')
    monkeypatch.setenv('OP_UPLOAD_MAX_BYTES_PER_REQUEST', '1000')
    monkeypatch.setenv('OP_UPLOAD_ACCEPTED_EXTENSIONS', '.jpg,.png')
    monkeypatch.setenv('OP_BATCH_MAX_ITEMS_PER_REQUEST', '9')
    monkeypatch.setenv('OP_REGION_DRAIN_POLL_INTERVAL_S', '5')
    monkeypatch.setenv('OP_REGION_DRAIN_STABLE_POLLS', '2')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)

    fake = AsyncMock()
    r = _client(fake).get('/curation/ingest/config')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['upload']['max_images_per_request'] == 7
    assert body['upload']['max_bytes_per_request'] == 1000
    assert body['upload']['accepted_extensions'] == ['.jpg', '.png']
    assert body['upload']['persists_bytes'] is True
    assert body['batch']['max_items'] == 9
    assert isinstance(body['batch']['source_roots'], list)
    assert body['region_drain']['poll_interval_s'] == 5
    assert body['region_drain']['stable_polls'] == 2


def test_region_drain_response_is_typed_with_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'aggregations': {
                'by_status': {
                    'buckets': [
                        {'key': 'pending_detection', 'doc_count': 3},
                        {'key': 'pending_verification', 'doc_count': 2},
                    ]
                }
            }
        }
    )
    r = _client(fake).get('/curation/ingest/region_drain')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['pending_detection'] == 3
    assert body['pending_verification'] == 2
    assert body['total_unfinished'] == 5
    assert body['drained'] is False
    assert 'stable_for_s' in body
    assert 'observed_at' in body


def test_region_drain_reports_drained_after_stable_zero_polls() -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'aggregations': {'by_status': {'buckets': []}}})
    client = _client(fake)
    last: dict[str, Any] = {}
    for _ in range(region_drain.region_drain_stable_polls() + 1):
        last = client.get('/curation/ingest/region_drain').json()
    assert last['total_unfinished'] == 0
    assert last['drained'] is True
