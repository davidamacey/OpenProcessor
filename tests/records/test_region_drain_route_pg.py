"""`GET /curation/ingest/region_drain`: the typed drain verdict and the dependency stall reason.

The verdict and the dependency timestamps live in the project's markers, so these run on a
database with the bound project backed by a real row.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation import region_drain


pytest_plugins = ['_records_world']
pytestmark = [pytest.mark.db, pytest.mark.usefixtures('world')]


def _client(fake: AsyncMock) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


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
    r = _client(fake).get('/curation/projects/default/ingest/region_drain')
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
        last = client.get('/curation/projects/default/ingest/region_drain').json()
    assert last['total_unfinished'] == 0
    assert last['drained'] is True


# =============================================================================
# V-1 -- region_dependencies / stall_reason on the drain response
# =============================================================================


def test_region_drain_reports_no_dependencies_with_no_active_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import DetectionProfile

    monkeypatch.setattr(
        'src.services.detection.profile_registry.region_profile_or_neutral',
        lambda: DetectionProfile(name='neutral', detector_model=''),
    )
    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'aggregations': {
                'by_status': {'buckets': [{'key': 'pending_detection', 'doc_count': 3}]}
            }
        }
    )
    r = _client(fake).get('/curation/projects/default/ingest/region_drain')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['region_dependencies'] == []
    assert body['stall_reason'] is None


def test_region_drain_surfaces_stall_reason_when_a_dependency_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import DetectionProfile
    from src.services.triton_control import TritonControlService

    monkeypatch.setattr(
        'src.services.detection.profile_registry.region_profile_or_neutral',
        lambda: DetectionProfile(name='active', detector_model='det_v1', segmenter_name=''),
    )

    async def _empty_index(self: TritonControlService) -> list[dict[str, str]]:
        return []

    monkeypatch.setattr(TritonControlService, 'get_repository_index', _empty_index)

    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'aggregations': {
                'by_status': {'buckets': [{'key': 'pending_detection', 'doc_count': 42}]}
            }
        }
    )
    r = _client(fake).get('/curation/projects/default/ingest/region_drain')
    assert r.status_code == 200, r.text
    body = r.json()
    [dep] = body['region_dependencies']
    assert dep['role'] == 'detector'
    assert dep['model'] == 'det_v1'
    assert dep['ready'] is False
    assert dep['unavailable_since'] is not None
    assert body['stall_reason'] is not None
    assert '42' in body['stall_reason']
    assert 'det_v1' in body['stall_reason']
