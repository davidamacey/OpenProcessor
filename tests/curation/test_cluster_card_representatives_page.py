"""``GET /clusters`` computes representatives only for the
requested ``[offset, offset+limit)`` page of the card list, via one
``_msearch`` (one body per cluster in the page) instead of a per-bucket
``top_hits`` sub-agg that decompressed stored ``_source`` for every
representative across every bucket."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.routers.curation.clusters import list_clusters


def _bucket(cid: int, size: int = 5) -> dict[str, Any]:
    return {
        'key': cid,
        'doc_count': size,
        'top_class': {'buckets': [{'key': 'van', 'doc_count': size}]},
        'labelled': {'doc_count': size},
        'validated': {'doc_count': 0},
        'subclusters': {'value': 0},
        'latest_update': {},
    }


def _msearch_response(n: int) -> dict[str, Any]:
    return {
        'responses': [
            {'hits': {'hits': [{'_id': f'crop-{i}', '_source': {'crop_id': f'crop-{i}'}}]}}
            for i in range(n)
        ]
    }


async def _call(
    os_client: AsyncMock,
    *,
    per_cluster: int = 4,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    return await list_clusters(
        os_client,
        per_cluster=per_cluster,
        max_clusters=100,
        kind='all',
        class_id=None,
        cluster_id=None,
        max_rank=None,
        min_blur_ratio=None,
        class_source=None,
        offset=offset,
        limit=limit,
    )


@pytest.mark.asyncio
async def test_terms_agg_has_no_top_hits_subagg() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': []}}})
    await _call(os_client)
    assert os_client.search.await_args is not None
    body = os_client.search.await_args.kwargs['body']
    assert 'top_hits' not in str(body['aggs']['clusters'])
    assert 'reps' not in body['aggs']['clusters']['aggs']


@pytest.mark.asyncio
async def test_msearch_only_issued_for_requested_page() -> None:
    buckets = [_bucket(cid) for cid in range(1, 6)]  # 5 clusters
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': buckets}}})
    os_client.msearch = AsyncMock(return_value=_msearch_response(2))

    resp = await _call(os_client, per_cluster=3, offset=1, limit=2)

    os_client.msearch.assert_awaited_once()
    assert os_client.msearch.await_args is not None
    msearch_body = os_client.msearch.await_args.kwargs['body']
    # 2 clusters on the page -> 2 header+query pairs.
    assert len(msearch_body) == 4
    queries = msearch_body[1::2]
    for q in queries:
        assert q['size'] == 3
        assert 'top_hits' not in str(q)

    # Cards outside [offset, offset+limit) keep empty representatives;
    # only the page's cards got filled.
    by_id = {c['cluster_id']: c for c in resp['items']}
    assert by_id[1]['representatives'] == []
    assert by_id[2]['representatives'] != []
    assert by_id[3]['representatives'] != []
    assert by_id[4]['representatives'] == []
    assert by_id[5]['representatives'] == []
    assert resp['representatives_offset'] == 1
    assert resp['representatives_limit'] == 2


@pytest.mark.asyncio
async def test_per_cluster_zero_skips_msearch_entirely() -> None:
    buckets = [_bucket(1)]
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': buckets}}})
    resp = await _call(os_client, per_cluster=0)
    os_client.msearch.assert_not_awaited()
    assert resp['items'][0]['representatives'] == []


@pytest.mark.asyncio
async def test_offset_beyond_card_count_yields_no_msearch() -> None:
    buckets = [_bucket(1)]
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': buckets}}})
    resp = await _call(os_client, per_cluster=4, offset=10, limit=5)
    os_client.msearch.assert_not_awaited()
    assert resp['items'][0]['representatives'] == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
