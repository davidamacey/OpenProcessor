"""Pre-0.4.0 items that have their vector get ``embedded`` recorded, so the
item wire never reads 'unknown' beside a stats count of embedded."""

from __future__ import annotations

from typing import Any

import pytest

from src.domain.embedding_state import legacy_embedded_clause
from src.services.projects.legacy_repair import backfill_embedded_state


class _Client:
    count_value = 7

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tasks = self
        self.polled: list[str] = []

    async def get(self, task_id: str) -> dict[str, Any]:
        self.polled.append(task_id)
        return {'completed': True, 'response': {'updated': 7}}

    async def count(self, **_kw: Any) -> dict[str, Any]:
        return {'count': self.count_value}

    async def update_by_query(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        return {'task': 'node:1'}


@pytest.mark.asyncio
async def test_backfill_targets_only_vectored_items_without_a_state() -> None:
    client = _Client()
    assert await backfill_embedded_state(client, 'items') == 7
    [call] = client.calls
    assert call['index'] == 'items'
    assert call['body']['query'] == legacy_embedded_clause()
    assert "'embedded'" in call['body']['script']['source']
    assert call['conflicts'] == 'proceed'
    assert call['wait_for_completion'] is False
    assert client.polled == ['node:1']
    must_not = call['body']['query']['bool']['must_not']
    assert must_not == [{'exists': {'field': 'embedding_state'}}]


@pytest.mark.asyncio
async def test_backfill_skips_without_a_task_when_nothing_is_legacy() -> None:
    client = _Client()
    client.count_value = 0
    assert await backfill_embedded_state(client, 'items') == 0
    assert client.calls == []


@pytest.mark.asyncio
async def test_backfill_is_throttled_and_single_sliced() -> None:
    client = _Client()
    await backfill_embedded_state(client, 'items')
    [call] = client.calls
    assert call['slices'] == 1
    assert 0 < call['requests_per_second'] <= 1000
