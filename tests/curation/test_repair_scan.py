"""scan_plans: paged search_after scan shared by the item repair jobs."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.repair_scan import scan_plans


class _PagedClient:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        assert index == 'items'
        self.bodies.append(body)
        start = 0
        if 'search_after' in body:
            start = self.ids.index(body['search_after'][0]) + 1
        page = self.ids[start : start + body['size']]
        hits = [{'_id': i, 'sort': [i], '_source': {'keep': i != 'b'}} for i in page]
        return {'hits': {'hits': hits}}


@pytest.mark.asyncio
async def test_pages_in_crop_id_order_and_drops_non_candidates() -> None:
    client = _PagedClient(['a', 'b', 'c', 'd', 'e'])
    out = await scan_plans(
        client,
        index='items',
        query={'match_all': {}},
        plan_one=lambda crop_id, src: crop_id if src['keep'] else None,
        page_size=2,
    )
    assert out == ['a', 'c', 'd', 'e']
    assert [b.get('search_after') for b in client.bodies] == [None, ['b'], ['d']]
    assert all(b['sort'] == [{'crop_id': 'asc'}] for b in client.bodies)
    assert all(b['_source'] == {'excludes': ['*embedding*']} for b in client.bodies)
