"""Tests for ``src/routers/curation/regions_fp.py`` (plan Wave 5 W5.c —
18.18% coverage, no prior test on any route).

Covers the read-side cluster-card assembly (``GET /regions/clusters``,
where the permanent false-positive bucket must sort first and carry
``cluster_kind='false_positive'``) and the "no centroids built yet"
short-circuit on ``GET /regions/suspected_false_positives`` — the two
routes cheaply testable against a fake OpenSearch without pulling in
the background-job machinery (``cluster_plates``,
``build_fp_centroids_endpoint``) or a real FAISS/embedding store.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import get_region_fields


F = get_region_fields()


class _FakeAggOS:
    def __init__(self, buckets: list[dict[str, Any]]) -> None:
        self._buckets = buckets

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'aggregations': {'clusters': {'buckets': self._buckets}}}


def _bucket(
    key: int, *, doc_count: int, rep_ids: list[str], n_sub: int = 0, validated: int = 0
) -> dict[str, Any]:
    return {
        'key': key,
        'doc_count': doc_count,
        'reps': {'hits': {'hits': [{'_id': rid} for rid in rep_ids]}},
        'subids': {'value': n_sub},
        'validated': {'doc_count': validated},
    }


@pytest.fixture
def app_client_factory():
    def _make(fake_os: Any) -> TestClient:
        from src.routers.curation import _raw_opensearch_dep, router as curation_router

        app = FastAPI()
        app.include_router(curation_router)
        app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
        return TestClient(app)

    return _make


def test_list_plate_clusters_pins_fp_bucket_first(app_client_factory: Any) -> None:
    from src.services.curation.clustering.orchestrator import FALSE_POSITIVE_REGION_CLUSTER_ID

    # A larger "good" bucket plus the (smaller) permanent FP bucket —
    # size ordering alone would put the good bucket first; the FP pin
    # must override that.
    buckets = [
        _bucket(7, doc_count=500, rep_ids=['crop-a']),
        _bucket(FALSE_POSITIVE_REGION_CLUSTER_ID, doc_count=3, rep_ids=['crop-fp']),
    ]
    client = app_client_factory(_FakeAggOS(buckets))

    resp = client.get('/curation/regions/clusters')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['count'] == 2
    assert body['clusters'][0]['id'] == FALSE_POSITIVE_REGION_CLUSTER_ID
    assert body['clusters'][0]['cluster_kind'] == 'false_positive'
    assert body['clusters'][1]['id'] == 7
    assert body['clusters'][1]['cluster_kind'] == 'candidate'


def test_list_plate_clusters_reports_representative_ids_and_subcluster_flag(
    app_client_factory: Any,
) -> None:
    buckets = [_bucket(9, doc_count=12, rep_ids=['crop-x', 'crop-y'], n_sub=3, validated=5)]
    client = app_client_factory(_FakeAggOS(buckets))

    resp = client.get('/curation/regions/clusters')
    assert resp.status_code == 200, resp.text
    cluster = resp.json()['clusters'][0]
    assert cluster['representative_crop_ids'] == ['crop-x', 'crop-y']
    assert cluster['has_subclusters'] is True
    assert cluster['n_subclusters'] == 3
    assert cluster['validated_count'] == 5


def test_list_plate_clusters_no_buckets_returns_empty(app_client_factory: Any) -> None:
    client = app_client_factory(_FakeAggOS([]))
    resp = client.get('/curation/regions/clusters')
    assert resp.status_code == 200
    assert resp.json() == {'clusters': [], 'count': 0}


def test_list_plate_clusters_surfaces_opensearch_error_as_503(app_client_factory: Any) -> None:
    class _BoomOS:
        async def search(self, **_kw: Any) -> dict[str, Any]:
            raise RuntimeError('cluster down')

    client = app_client_factory(_BoomOS())
    resp = client.get('/curation/regions/clusters')
    assert resp.status_code == 503


def test_suspected_false_positives_short_circuits_when_no_centroids_built(
    app_client_factory: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    # Point the store at an empty tmp dir so .load() genuinely finds nothing,
    # rather than depending on this host's real state directory contents.
    monkeypatch.setattr(
        FalsePositiveCentroidStore, '__init__', lambda self: setattr(self, 'metadata', {})
    )
    monkeypatch.setattr(FalsePositiveCentroidStore, 'load', lambda _self: False)

    client = app_client_factory(_FakeAggOS([]))
    resp = client.get('/curation/regions/suspected_false_positives')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['items'] == []
    assert body['total'] == 0
    assert body['centroids_built'] is False


def test_plate_cluster_status_and_fp_centroid_status_are_reachable(
    app_client_factory: Any,
) -> None:
    client = app_client_factory(_FakeAggOS([]))
    resp = client.get('/curation/regions/cluster/status')
    assert resp.status_code == 200
    resp2 = client.get('/curation/regions/fp_centroids/status')
    assert resp2.status_code == 200
