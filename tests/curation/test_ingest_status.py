"""``GET /curation/ingest/status`` (F-13).

Two bugs fixed together: the request was missing ``track_total_hits``
(OpenSearch silently caps the reported ``total`` at 10000 once the images
index exceeds it), and the 14-day recent-activity histogram ran a
``date_histogram`` over *every day ever ingested* only to keep the first
14 desc-sorted buckets in Python -- moved into a ``range`` filter agg so
OpenSearch only bucket-computes the last 14 days.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    import pytest


def _client(
    monkeypatch: pytest.MonkeyPatch, search_resp: dict[str, Any]
) -> tuple[TestClient, AsyncMock]:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake = AsyncMock()
    fake.search = AsyncMock(return_value=search_resp)
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app), fake


_RESP = {
    'hits': {'total': {'value': 42}},
    'aggregations': {
        'by_source': {'buckets': [{'key': 'hd1', 'doc_count': 10}]},
        'by_day': {
            'doc_count': 30,
            'days': {
                'buckets': [
                    {'key_as_string': '2026-09-24', 'doc_count': 5},
                    {'key_as_string': '2026-09-23', 'doc_count': 7},
                ]
            },
        },
    },
}


def test_ingest_status_request_tracks_total_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    client, fake = _client(monkeypatch, _RESP)
    r = client.get('/curation/ingest/status')
    assert r.status_code == 200, r.text

    body = fake.search.call_args.kwargs['body']
    assert body['track_total_hits'] is True


def test_ingest_status_by_day_uses_a_range_filter_not_a_python_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _client(monkeypatch, _RESP)
    r = client.get('/curation/ingest/status')
    assert r.status_code == 200, r.text

    aggs = fake.search.call_args.kwargs['body']['aggs']
    by_day = aggs['by_day']
    # The date_histogram must be nested inside a 14-day range filter, not
    # run unbounded over the whole index.
    assert 'filter' in by_day
    assert by_day['filter'] == {'range': {'indexed_at': {'gte': 'now-14d/d'}}}
    assert 'date_histogram' in by_day['aggs']['days']


def test_ingest_status_response_shape_unwraps_the_filtered_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _fake = _client(monkeypatch, _RESP)
    r = client.get('/curation/ingest/status')
    body = r.json()
    assert body['total'] == 42
    assert body['by_source'] == [{'key': 'hd1', 'doc_count': 10}]
    assert body['by_day'] == [
        {'key_as_string': '2026-09-24', 'doc_count': 5},
        {'key_as_string': '2026-09-23', 'doc_count': 7},
    ]
