"""Pre-0.4.0 items that have their vector get ``embedded`` recorded, so the
item wire never reads 'unknown' beside a stats count of embedded."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.embedding_state import backfill_embedded_state, legacy_embedded_clause


class _Client:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def update_by_query(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        return {'updated': 7}


@pytest.mark.asyncio
async def test_backfill_targets_only_vectored_items_without_a_state() -> None:
    client = _Client()
    assert await backfill_embedded_state(client, 'items') == 7
    [call] = client.calls
    assert call['index'] == 'items'
    assert call['body']['query'] == legacy_embedded_clause()
    assert "'embedded'" in call['body']['script']['source']
    assert call['conflicts'] == 'proceed'
    must_not = call['body']['query']['bool']['must_not']
    assert must_not == [{'exists': {'field': 'embedding_state'}}]
