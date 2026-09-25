"""
Unit tests for the curation clustering router's
``GET /curation/clusters/representatives`` endpoint. Mocks the
OpenSearch dependency and asserts the response shape.

Representatives no longer come from a per-bucket
``top_hits`` sub-agg — the ``terms`` agg only returns cluster ids, and
one ``_msearch`` (one body per cluster in the requested page) fetches
representatives for that page only.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def fake_opensearch() -> AsyncMock:
    fake = AsyncMock()
    fake.indices = AsyncMock()
    fake.indices.exists = AsyncMock(return_value=True)
    fake.indices.create = AsyncMock(return_value={'acknowledged': True})
    fake.indices.refresh = AsyncMock(return_value={'_shards': {}})
    fake.search = AsyncMock(return_value={'hits': {'hits': []}, 'aggregations': {}})
    fake.msearch = AsyncMock(return_value={'responses': []})
    return fake


@pytest.fixture
def fake_triton_pool() -> AsyncMock:
    fake = AsyncMock()
    fake.health_check = AsyncMock(return_value=True)
    return fake


@pytest.fixture
def app_client(fake_opensearch: AsyncMock, fake_triton_pool: AsyncMock):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.core.dependencies import get_async_triton
    from src.routers.curation._common import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    # The curation router unwraps via ``_raw_opensearch_dep`` (it calls
    # ``get_opensearch()`` directly rather than depending on it), so the
    # override has to target the unwrap dep.
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch
    app.dependency_overrides[get_async_triton] = lambda: fake_triton_pool

    with TestClient(app) as client:
        yield client


def _cluster_id_buckets(*cluster_ids: int) -> dict[str, Any]:
    return {
        'aggregations': {
            'clusters': {'buckets': [{'key': cid, 'doc_count': 1} for cid in cluster_ids]}
        },
    }


def _msearch_hit(crop_id: str, distance: float, class_name: str | None) -> dict[str, Any]:
    return {
        '_id': crop_id,
        '_source': {
            'crop_id': crop_id,
            'cluster_distance': distance,
            'class_name': class_name,
        },
    }


def test_cluster_representatives_returns_keyed_dict(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    fake_opensearch.search = AsyncMock(return_value=_cluster_id_buckets(1, 42))
    fake_opensearch.msearch = AsyncMock(
        return_value={
            'responses': [
                {
                    'hits': {
                        'hits': [
                            _msearch_hit('c1-a', 0.10, 'class_b'),
                            _msearch_hit('c1-b', 0.12, 'class_b'),
                            _msearch_hit('c1-c', 0.15, None),
                        ],
                    },
                },
                {
                    'hits': {
                        'hits': [
                            _msearch_hit('c42-a', 0.05, 'sportycar'),
                            _msearch_hit('c42-b', 0.07, 'sportycar'),
                            _msearch_hit('c42-c', 0.09, 'pickup'),
                        ],
                    },
                },
            ],
        },
    )

    r = app_client.get('/curation/clusters/representatives')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['count'] == 2
    clusters = body['clusters']

    # Cluster ids are stringified.
    assert set(clusters.keys()) == {'1', '42'}

    crops_1 = clusters['1']
    assert [c['crop_id'] for c in crops_1] == ['c1-a', 'c1-b', 'c1-c']
    assert [c['class_name'] for c in crops_1] == ['class_b', 'class_b', None]
    assert crops_1[0]['cluster_distance'] == 0.10

    crops_42 = clusters['42']
    assert [c['crop_id'] for c in crops_42] == ['c42-a', 'c42-b', 'c42-c']
    assert [c['class_name'] for c in crops_42] == ['sportycar', 'sportycar', 'pickup']


def test_cluster_representatives_query_shape_has_no_top_hits(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    """The terms agg must no longer carry a top_hits sub-agg."""
    fake_opensearch.search = AsyncMock(return_value=_cluster_id_buckets())

    r = app_client.get('/curation/clusters/representatives?per_cluster=3&max_clusters=50')
    assert r.status_code == 200, r.text

    fake_opensearch.search.assert_awaited_once()
    assert fake_opensearch.search.await_args is not None
    body = fake_opensearch.search.await_args.kwargs['body']
    assert fake_opensearch.search.await_args.kwargs['index'] == 'op_items'
    assert body['size'] == 0

    terms = body['aggs']['clusters']['terms']
    assert terms['field'] == 'cluster_id'
    assert terms['size'] == 50  # offset(0) + max_clusters(50)
    assert 'top_hits' not in str(body['aggs']['clusters'])
    assert 'aggs' not in body['aggs']['clusters']

    fake_opensearch.msearch.assert_not_awaited()  # no cluster ids -> no msearch


def test_cluster_representatives_msearch_only_covers_the_page(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    fake_opensearch.search = AsyncMock(return_value=_cluster_id_buckets(1, 42))
    fake_opensearch.msearch = AsyncMock(return_value={'responses': [{}, {}]})

    r = app_client.get('/curation/clusters/representatives?per_cluster=3')
    assert r.status_code == 200, r.text

    fake_opensearch.msearch.assert_awaited_once()
    assert fake_opensearch.msearch.await_args is not None
    msearch_body = fake_opensearch.msearch.await_args.kwargs['body']
    # One header + one query per cluster in the page.
    assert len(msearch_body) == 4
    headers = msearch_body[0::2]
    queries = msearch_body[1::2]
    assert all(h == {'index': 'op_items'} for h in headers)
    for q in queries:
        assert 'top_hits' not in str(q)
        assert q['size'] == 3
        assert q['query']['bool']['filter'] == [{'term': {'cluster_id': 1}}] or q['query']['bool'][
            'filter'
        ] == [{'term': {'cluster_id': 42}}]
        assert q['query']['bool']['must_not'] == [{'term': {'class_excluded': True}}]
        assert q['sort'] == [
            {'cluster_distance': {'order': 'asc', 'missing': '_last', 'unmapped_type': 'double'}},
            {'crop_id': 'asc'},
        ]
        # cluster_id + cluster_distance_cluster_id let a stale distance be
        # nulled.
        assert q['_source'] == [
            'crop_id',
            'cluster_id',
            'cluster_distance',
            'cluster_distance_cluster_id',
            'class_name',
            'cluster_subid',
        ]


def test_cluster_representatives_offset_limits_page_size(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    fake_opensearch.search = AsyncMock(return_value=_cluster_id_buckets(1, 2, 3))
    fake_opensearch.msearch = AsyncMock(return_value={'responses': [{}]})

    r = app_client.get('/curation/clusters/representatives?offset=2&max_clusters=1')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['offset'] == 2
    assert set(body['clusters'].keys()) == {'3'}

    assert fake_opensearch.search.await_args is not None
    terms = fake_opensearch.search.await_args.kwargs['body']['aggs']['clusters']['terms']
    assert terms['size'] == 3  # offset(2) + max_clusters(1)


def test_cluster_representatives_excludes_class_excluded_items(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    """Excluded items must not surface as cluster representatives --
    this endpoint had no class_excluded guard at all before."""
    fake_opensearch.search = AsyncMock(return_value=_cluster_id_buckets())

    r = app_client.get('/curation/clusters/representatives')
    assert r.status_code == 200, r.text

    assert fake_opensearch.search.await_args is not None
    body = fake_opensearch.search.await_args.kwargs['body']
    assert body['query'] == {
        'bool': {'filter': [], 'must_not': [{'term': {'class_excluded': True}}]}
    }


def test_cluster_representatives_class_id_filter_keeps_class_excluded_guard(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    fake_opensearch.search = AsyncMock(return_value=_cluster_id_buckets())

    r = app_client.get('/curation/clusters/representatives?class_id=7')
    assert r.status_code == 200, r.text

    assert fake_opensearch.search.await_args is not None
    body = fake_opensearch.search.await_args.kwargs['body']
    assert body['query'] == {
        'bool': {
            'filter': [{'term': {'class_id': 7}}],
            'must_not': [{'term': {'class_excluded': True}}],
        }
    }


def test_cluster_representatives_handles_empty_buckets(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    fake_opensearch.search = AsyncMock(return_value=_cluster_id_buckets())
    r = app_client.get('/curation/clusters/representatives')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['clusters'] == {}
    assert body['count'] == 0


def test_cluster_representatives_falls_back_to_doc_id(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    """If a hit's _source omits crop_id, the response uses the OpenSearch _id."""
    fake_opensearch.search = AsyncMock(return_value=_cluster_id_buckets(5))
    fake_opensearch.msearch = AsyncMock(
        return_value={
            'responses': [
                {
                    'hits': {
                        'hits': [
                            {
                                '_id': 'os-doc-id-9',
                                '_source': {'cluster_distance': 0.2, 'class_name': 'pickup'},
                            },
                        ],
                    },
                },
            ],
        }
    )
    r = app_client.get('/curation/clusters/representatives')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['clusters']['5'][0]['crop_id'] == 'os-doc-id-9'


def test_cluster_representatives_500_on_opensearch_error(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    fake_opensearch.search = AsyncMock(side_effect=RuntimeError('boom'))
    r = app_client.get('/curation/clusters/representatives')
    assert r.status_code == 500
    assert 'representatives query failed' in r.text


def test_cluster_representatives_500_on_msearch_error(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    fake_opensearch.search = AsyncMock(return_value=_cluster_id_buckets(1))
    fake_opensearch.msearch = AsyncMock(side_effect=RuntimeError('boom'))
    r = app_client.get('/curation/clusters/representatives')
    assert r.status_code == 500
    assert 'representatives msearch failed' in r.text
