"""``GET /classes`` takes its counts from the items index, never from a stored field (02a section 3).

The route runs one terms aggregation on ``class_id`` (with a validated filter beneath it) and joins
the buckets onto the registry's classes; a class with no bucket has zero.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from _class_registry_fake import InMemoryRegistry, use_registry
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.curation_opensearch import RegistryClassEntry


def test_a_registry_class_stores_no_counts() -> None:
    assert 'sample_count' not in RegistryClassEntry.model_fields
    assert 'validated_count' not in RegistryClassEntry.model_fields


@pytest.fixture
def client() -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router

    use_registry(InMemoryRegistry(names=['sedan', 'suv', 'van']))
    aggs = {
        'by_class': {
            'buckets': [
                {
                    'key': 0,
                    'doc_count': 7,
                    'validated': {'doc_count': 3},
                    'validated_test_holdout': {'doc_count': 1},
                    'validated_excluded': {'doc_count': 0},
                },
                {'key': 1, 'doc_count': 2, 'validated': {'doc_count': 0}},
            ]
        },
        'by_cluster': {'classes': {'buckets': [{'key': 0, 'doc_count': 9}]}},
    }
    os = AsyncMock()
    os.search = AsyncMock(return_value={'aggregations': aggs})
    os.count = AsyncMock(return_value={'count': 0})
    app = FastAPI()
    mount_curation_routers(app, router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: os
    c = TestClient(app)
    c.os = os  # type: ignore[attr-defined]
    return c


def test_counts_come_from_one_aggregation_and_missing_buckets_are_zero(client: Any) -> None:
    body = client.get('/curation/projects/default/classes').json()
    by_name: dict[str, Any] = {c['class_name']: c for c in body['classes']}
    assert (by_name['sedan']['sample_count'], by_name['sedan']['validated_count']) == (7, 3)
    assert by_name['sedan']['cluster_size'] == 9
    assert by_name['sedan']['trainable'] == 2
    assert (by_name['suv']['sample_count'], by_name['suv']['validated_count']) == (2, 0)
    assert (by_name['van']['sample_count'], by_name['van']['validated_count']) == (0, 0)
    assert client.os.search.await_count == 1
