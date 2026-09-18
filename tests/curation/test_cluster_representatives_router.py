"""
Unit tests for the curation clustering router's
``GET /curation/clusters/representatives`` endpoint. Mocks the
OpenSearch dependency and asserts the response shape.
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


def _bucket(cluster_id: int, hits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'key': cluster_id,
        'doc_count': len(hits),
        'reps': {'hits': {'hits': hits}},
    }


def _hit(crop_id: str, distance: float, class_name: str | None) -> dict[str, Any]:
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
    fake_opensearch.search = AsyncMock(
        return_value={
            'aggregations': {
                'clusters': {
                    'buckets': [
                        _bucket(
                            1,
                            [
                                _hit('c1-a', 0.10, 'cruiserbike'),
                                _hit('c1-b', 0.12, 'cruiserbike'),
                                _hit('c1-c', 0.15, None),
                            ],
                        ),
                        _bucket(
                            42,
                            [
                                _hit('c42-a', 0.05, 'sportycar'),
                                _hit('c42-b', 0.07, 'sportycar'),
                                _hit('c42-c', 0.09, 'pickup'),
                            ],
                        ),
                    ],
                },
            },
        }
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
    assert [c['class_name'] for c in crops_1] == ['cruiserbike', 'cruiserbike', None]
    assert crops_1[0]['cluster_distance'] == 0.10

    crops_42 = clusters['42']
    assert [c['crop_id'] for c in crops_42] == ['c42-a', 'c42-b', 'c42-c']
    assert [c['class_name'] for c in crops_42] == ['sportycar', 'sportycar', 'pickup']


def test_cluster_representatives_query_shape(app_client: Any, fake_opensearch: AsyncMock) -> None:
    """Verify the OpenSearch body uses the expected aggregation."""
    fake_opensearch.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': []}}})

    r = app_client.get('/curation/clusters/representatives?per_cluster=3&max_clusters=50')
    assert r.status_code == 200, r.text

    fake_opensearch.search.assert_awaited_once()
    assert fake_opensearch.search.await_args is not None
    body = fake_opensearch.search.await_args.kwargs['body']
    assert fake_opensearch.search.await_args.kwargs['index'] == 'op_items'
    assert body['size'] == 0

    terms = body['aggs']['clusters']['terms']
    assert terms['field'] == 'cluster_id'
    assert terms['size'] == 50

    top_hits = body['aggs']['clusters']['aggs']['reps']['top_hits']
    assert top_hits['size'] == 3
    assert top_hits['sort'] == [
        {'cluster_distance': {'order': 'asc', 'missing': '_last', 'unmapped_type': 'double'}}
    ]
    assert top_hits['_source'] == ['crop_id', 'cluster_distance', 'class_name', 'cluster_subid']


def test_cluster_representatives_handles_empty_buckets(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    fake_opensearch.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': []}}})
    r = app_client.get('/curation/clusters/representatives')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {'clusters': {}, 'count': 0}


def test_cluster_representatives_falls_back_to_doc_id(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    """If a hit's _source omits crop_id, the response uses the OpenSearch _id."""
    fake_opensearch.search = AsyncMock(
        return_value={
            'aggregations': {
                'clusters': {
                    'buckets': [
                        {
                            'key': 5,
                            'doc_count': 1,
                            'reps': {
                                'hits': {
                                    'hits': [
                                        {
                                            '_id': 'os-doc-id-9',
                                            '_source': {
                                                'cluster_distance': 0.2,
                                                'class_name': 'pickup',
                                            },
                                        },
                                    ],
                                },
                            },
                        },
                    ],
                },
            },
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
