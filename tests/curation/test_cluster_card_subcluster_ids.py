"""``GET /clusters`` serves each card's sub-cluster ids, so a client never has to guess the
tab list from the page of crops it happens to have loaded."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.routers.curation.clusters import list_clusters
from src.services.curation.item_filter import ItemFilter


def _bucket(cid: int, subids: list[str] | None) -> dict[str, Any]:
    bucket: dict[str, Any] = {
        'key': cid,
        'doc_count': 9,
        'top_class': {'buckets': []},
        'labelled': {'doc_count': 0},
        'validated': {'doc_count': 0},
        'subclusters': {'value': len(subids or [])},
        'latest_update': {},
    }
    if subids is not None:
        bucket['subcluster_ids'] = {'buckets': [{'key': s, 'doc_count': 1} for s in subids]}
    return bucket


async def _list(*buckets: dict[str, Any]) -> tuple[dict[str, Any], AsyncMock]:
    os_client = AsyncMock()
    os_client.search = AsyncMock(
        return_value={'aggregations': {'clusters': {'buckets': list(buckets)}}}
    )
    resp = await list_clusters(
        os_client,
        per_cluster=0,
        max_clusters=100,
        kind='all',
        cluster_id=None,
        item_filter=ItemFilter(),
        min_blur_ratio=None,
        class_source=None,
    )
    return resp, os_client


@pytest.mark.asyncio
async def test_each_card_lists_its_subcluster_ids_in_order() -> None:
    resp, _ = await _list(_bucket(7, ['7a', '7b', '7aa']), _bucket(8, None))
    cards = {c['cluster_id']: c for c in resp['items']}
    assert cards[7]['subcluster_ids'] == ['7a', '7aa', '7b']
    assert cards[8]['subcluster_ids'] == []


@pytest.mark.asyncio
async def test_the_aggregation_asks_for_the_ids_by_key() -> None:
    _, os_client = await _list(_bucket(7, []))
    aggs = os_client.search.call_args.kwargs['body']['aggs']['clusters']['aggs']
    assert aggs['subcluster_ids']['terms']['field'] == 'cluster_subid'
    assert aggs['subcluster_ids']['terms']['order'] == {'_key': 'asc'}
