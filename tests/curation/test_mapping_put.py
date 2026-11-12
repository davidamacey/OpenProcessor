"""put_properties / put_properties_one_by_one: the result dicts the ensure_* migrations return."""

from __future__ import annotations

from typing import Any

import pytest
from opensearchpy.exceptions import RequestError

from src.clients.curation_opensearch.mapping_put import put_properties, put_properties_one_by_one


class _Indices:
    def __init__(self, fail: dict[str, Exception] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail or {}

    async def put_mapping(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({'index': index, **body})
        for field, exc in self.fail.items():
            if field in body['properties']:
                raise exc
        return {'acknowledged': True}


class _Client:
    def __init__(self, fail: dict[str, Exception] | None = None) -> None:
        self.indices = _Indices(fail)


@pytest.mark.asyncio
async def test_put_properties_acknowledged() -> None:
    client = _Client()
    out = await put_properties(client, 'ix', {'a': {'type': 'keyword'}, 'b': {'type': 'date'}})  # type: ignore[arg-type]
    assert out == {'acknowledged': True, 'index': 'ix', 'fields_added': ['a', 'b']}
    assert len(client.indices.calls) == 1


@pytest.mark.asyncio
async def test_put_properties_reports_an_opensearch_failure_without_raising() -> None:
    client = _Client({'a': RequestError(400, 'boom', {})})
    out = await put_properties(client, 'ix', {'a': {'type': 'keyword'}})  # type: ignore[arg-type]
    assert out['acknowledged'] is False
    assert out['fields_added'] == ['a']
    assert 'boom' in out['error']


@pytest.mark.asyncio
async def test_one_by_one_stops_at_an_unrecoverable_failure_and_keeps_the_added_fields() -> None:
    client = _Client({'b': RequestError(400, 'boom', {})})
    out = await put_properties_one_by_one(
        client,  # type: ignore[arg-type]
        'ix',
        {'a': {'type': 'keyword'}, 'b': {'type': 'date'}, 'c': {'type': 'date'}},
    )
    assert out['acknowledged'] is False
    assert out['fields_added'] == ['a']
    assert len(client.indices.calls) == 2


@pytest.mark.asyncio
async def test_one_by_one_adds_every_field_when_all_succeed() -> None:
    client = _Client()
    out = await put_properties_one_by_one(client, 'ix', {'a': {}, 'b': {}})  # type: ignore[arg-type]
    assert out == {
        'acknowledged': True,
        'index': 'ix',
        'fields_added': ['a', 'b'],
        'conflicts': [],
    }
