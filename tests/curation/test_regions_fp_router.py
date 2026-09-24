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


class _FakeFpSearchOS:
    """Fake OS for the suspected-FP scoring path (F-1): one page of embedding
    hits via ``search``/``scroll``, then ``mget`` to hydrate item fields for
    the scored page. Records the exact ``mget`` kwargs so the test can assert
    the fix uses ``_source_excludes=`` rather than the broken ``_source={...}``
    form (opensearch-py stringifies a dict ``_source`` into the query param,
    which OpenSearch then reads as an include pattern matching nothing).
    """

    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding
        self.mget_calls: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {
            '_scroll_id': 'scroll-1',
            'hits': {'hits': [{'_id': 'crop-fp-1', '_source': {F.embedding: self._embedding}}]},
        }

    async def scroll(self, *, scroll_id: str, scroll: str) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': None, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str) -> None:  # noqa: ARG002
        return None

    async def mget(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        self.mget_calls.append({'index': index, 'body': body, **kw})
        return {
            'docs': [
                {
                    '_id': cid,
                    'found': True,
                    '_source': {'image_path': '/x.jpg', 'class_name': 'thing'},
                }
                for cid in body['ids']
            ]
        }


def test_suspected_false_positives_mget_uses_source_excludes_kwarg(
    app_client_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-1 regression: the mget call must pass ``_source_excludes=`` (a real
    opensearch-py kwarg), not ``_source={'excludes': [...]}`` (silently
    stringified into a useless include pattern -> every item comes back with
    an empty ``_source``).
    """
    import numpy as np

    from src.routers.curation.regions import _REGION_SOURCE_EXCLUDES
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    def _fake_init(self: Any) -> None:
        self.metadata = {'subids': ['a']}

    monkeypatch.setattr(FalsePositiveCentroidStore, '__init__', _fake_init)
    monkeypatch.setattr(FalsePositiveCentroidStore, 'load', lambda _self: True)
    monkeypatch.setattr(
        FalsePositiveCentroidStore,
        'search',
        lambda _self, embs: (np.zeros(len(embs), dtype=np.float32), np.zeros(len(embs), dtype=int)),
    )

    fake_os = _FakeFpSearchOS(embedding=[0.1] * 8)
    client = app_client_factory(fake_os)

    resp = client.get('/curation/regions/suspected_false_positives')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['centroids_built'] is True
    assert body['total'] == 1

    assert len(fake_os.mget_calls) == 1
    call = fake_os.mget_calls[0]
    assert '_source_excludes' in call
    assert call['_source_excludes'] == _REGION_SOURCE_EXCLUDES
    assert '_source' not in call

    # And the fix actually restores non-empty item fields end-to-end.
    assert body['items'][0]['image_path'] == '/x.jpg'


class _FakeScrollOS:
    """Fake covering the scroll surface ``suspected_false_positives`` uses:
    ``search`` (scroll-open), ``scroll`` (paging), ``clear_scroll``, ``mget``."""

    def __init__(
        self, hits_pages: list[list[dict[str, Any]]], *, raise_on_scroll: bool = False
    ) -> None:
        self.hits_pages = hits_pages
        self.raise_on_scroll = raise_on_scroll
        self.search_calls = 0
        self.scroll_calls = 0
        self.clear_scroll_calls = 0

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        self.search_calls += 1
        page = self.hits_pages[0] if self.hits_pages else []
        return {'_scroll_id': 'sid-0', 'hits': {'hits': page}}

    async def scroll(self, *, scroll_id: str, scroll: str) -> dict[str, Any]:  # noqa: ARG002
        self.scroll_calls += 1
        if self.raise_on_scroll:
            msg = 'transport boom mid-scroll'
            raise RuntimeError(msg)
        if self.scroll_calls < len(self.hits_pages):
            return {
                '_scroll_id': f'sid-{self.scroll_calls}',
                'hits': {'hits': self.hits_pages[self.scroll_calls]},
            }
        return {'_scroll_id': None, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str) -> None:  # noqa: ARG002
        self.clear_scroll_calls += 1

    async def mget(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'docs': [{'_id': i, 'found': True, '_source': {}} for i in body['ids']]}


def _patch_fp_store(monkeypatch: pytest.MonkeyPatch, *, trained_at: str = 'T1') -> None:
    import numpy as np

    from src.services.detection.fp_store import FalsePositiveCentroidStore

    monkeypatch.setattr(
        FalsePositiveCentroidStore,
        '__init__',
        lambda self: setattr(self, 'metadata', {'trained_at': trained_at, 'subids': ['s1']}),
    )
    monkeypatch.setattr(FalsePositiveCentroidStore, 'load', lambda _self: True)
    monkeypatch.setattr(
        FalsePositiveCentroidStore,
        'search',
        lambda _self, embs: (np.zeros(len(embs), dtype=np.float32), np.zeros(len(embs), dtype=int)),
    )


def test_suspected_fp_second_page_within_ttl_does_not_rescroll(
    app_client_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-18: the scored-list cache means a page-2 request shortly after
    page-1 doesn't re-scroll the whole region-embedding pool."""
    from src.routers.curation import regions_fp as regions_fp_mod

    regions_fp_mod._suspected_fp_cache.clear()
    _patch_fp_store(monkeypatch)

    hits = [{'_id': f'r{i}', '_source': {F.embedding: [0.1, 0.2, 0.3, 0.4]}} for i in range(3)]
    os_fake = _FakeScrollOS([hits])
    client = app_client_factory(os_fake)

    r1 = client.get('/curation/regions/suspected_false_positives?page=1&page_size=2')
    assert r1.status_code == 200, r1.text
    r2 = client.get('/curation/regions/suspected_false_positives?page=2&page_size=2')
    assert r2.status_code == 200, r2.text

    assert os_fake.search_calls == 1


def test_suspected_fp_scroll_exception_still_clears_scroll(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-18: an exception mid-scroll must still hit clear_scroll (finally),
    not leak an open scroll context."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.routers.curation import (
        _raw_opensearch_dep,
        regions_fp as regions_fp_mod,
        router as curation_router,
    )

    regions_fp_mod._suspected_fp_cache.clear()
    _patch_fp_store(monkeypatch, trained_at='T2')

    hits_page_1 = [{'_id': 'r0', '_source': {F.embedding: [0.1, 0.2, 0.3, 0.4]}}]
    os_fake = _FakeScrollOS([hits_page_1, []], raise_on_scroll=True)

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: os_fake

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get('/curation/regions/suspected_false_positives')

    assert resp.status_code == 500
    assert os_fake.clear_scroll_calls == 1
