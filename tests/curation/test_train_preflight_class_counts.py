"""F-28.3: preflight's per-class validated + test-holdout counts merge
into one ``_search`` (two sibling filter aggs) instead of two separate
round trips."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.routers.curation_train import _count_validated_and_test_per_class


def _agg_response(validated: dict[int, int], test: dict[int, int]) -> dict[str, Any]:
    return {
        'aggregations': {
            'validated_by_class': {
                'by_class': {'buckets': [{'key': k, 'doc_count': v} for k, v in validated.items()]}
            },
            'test_by_class': {
                'by_class': {'buckets': [{'key': k, 'doc_count': v} for k, v in test.items()]}
            },
        }
    }


@pytest.mark.asyncio
async def test_issues_exactly_one_search() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value=_agg_response({1: 10, 2: 3}, {1: 5, 2: 1}))

    validated, test = await _count_validated_and_test_per_class(os_client, [1, 2])

    os_client.search.assert_awaited_once()
    assert validated == {1: 10, 2: 3}
    assert test == {1: 5, 2: 1}


@pytest.mark.asyncio
async def test_query_body_has_two_sibling_filter_aggs_sharing_class_scope() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value=_agg_response({}, {}))

    await _count_validated_and_test_per_class(os_client, [7, 8])

    assert os_client.search.await_args is not None
    body = os_client.search.await_args.kwargs['body']
    assert body['size'] == 0
    assert body['query'] == {'bool': {'filter': [{'terms': {'class_id': [7, 8]}}]}}
    aggs = body['aggs']
    assert set(aggs) == {'validated_by_class', 'test_by_class'}
    assert aggs['validated_by_class']['filter'] == {'term': {'class_validated': True}}
    assert aggs['test_by_class']['filter'] == {
        'bool': {'filter': [{'term': {'class_validated': True}}, {'term': {'test_holdout': True}}]}
    }
    for agg_name in ('validated_by_class', 'test_by_class'):
        assert aggs[agg_name]['aggs']['by_class']['terms']['field'] == 'class_id'


@pytest.mark.asyncio
async def test_missing_class_defaults_to_zero_in_both_dicts() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value=_agg_response({1: 10}, {}))

    validated, test = await _count_validated_and_test_per_class(os_client, [1, 2])

    assert validated == {1: 10, 2: 0}
    assert test == {1: 0, 2: 0}


@pytest.mark.asyncio
async def test_empty_class_ids_short_circuits_without_a_query() -> None:
    os_client = AsyncMock()
    validated, test = await _count_validated_and_test_per_class(os_client, [])
    os_client.search.assert_not_awaited()
    assert validated == {}
    assert test == {}


@pytest.mark.asyncio
async def test_opensearch_failure_returns_zeros_for_every_class() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(side_effect=RuntimeError('boom'))
    validated, test = await _count_validated_and_test_per_class(os_client, [1, 2])
    assert validated == {1: 0, 2: 0}
    assert test == {1: 0, 2: 0}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
