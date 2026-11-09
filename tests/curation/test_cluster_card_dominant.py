"""``GET /clusters`` cards: dominant class, label agreement and the aggregation query."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from curation._cluster_cards_helpers import list_cards, uid


def _bucket(
    cid: int, size: int, classes: list[tuple[str, int]], labelled: int | None = None
) -> dict[str, Any]:
    return {
        'key': uid(cid),
        'doc_count': size,
        'top_class': {'buckets': [{'key': k, 'doc_count': n} for k, n in classes[:3]]},
        'labelled': {'doc_count': sum(n for _k, n in classes) if labelled is None else labelled},
        'validated': {'doc_count': 0},
        'latest_update': {},
    }


async def _cards(*buckets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    os_client = AsyncMock()
    os_client.search = AsyncMock(
        return_value={'aggregations': {'clusters': {'buckets': list(buckets)}}}
    )
    resp = await list_cards(os_client, per_cluster=0, max_clusters=100)
    return {c['cluster_uid']: c for c in resp['items']}


@pytest.mark.asyncio
async def test_the_dominant_class_is_the_top_labelled_class() -> None:
    card = (await _cards(_bucket(1, 116, [('dumptruck', 1), ('van', 1), ('bus', 1)])))[uid(1)]
    assert card['dominant_class_name'] == 'dumptruck'
    assert card['dominant_count'] == 1
    assert card['labelled_count'] == 3


@pytest.mark.asyncio
async def test_labelled_count_includes_classes_beyond_top_buckets() -> None:
    classes = [('van', 4), ('bus', 1), ('car', 1), ('suv', 1), ('cab', 1)]
    card = (await _cards(_bucket(2, 40, classes)))[uid(2)]
    assert card['labelled_count'] == 8
    # The top class's share of the labels is label_agreement.
    assert card['label_agreement'] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_label_agreement_null_when_unlabelled() -> None:
    card = (await _cards(_bucket(3, 10, [])))[uid(3)]
    assert card['label_agreement'] is None
    assert card['dominant_class_name'] is None
    assert card['is_unlabeled'] is True


@pytest.mark.asyncio
async def test_the_query_groups_by_the_served_cluster_uid() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': []}}})
    await list_cards(os_client, per_cluster=0, max_clusters=100)
    body = os_client.search.await_args.kwargs['body']
    assert body['aggs']['clusters']['terms']['field'] == 'cluster_uid'
    assert {'exists': {'field': 'cluster_uid'}} in body['query']['bool']['filter']
    assert 'cluster_id' not in str(body)


@pytest.mark.asyncio
async def test_one_cluster_is_asked_for_by_uid() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': []}}})
    await list_cards(os_client, per_cluster=0, max_clusters=100, cluster_uid=uid(4))
    body = os_client.search.await_args.kwargs['body']
    assert {'term': {'cluster_uid': uid(4)}} in body['query']['bool']['filter']
