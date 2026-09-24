"""F-16 coverage for ``src.services.curation.selection.pool_fetch``.

Count-first cap check: a pool already known (via ``count``) to exceed
``cap`` must never be scrolled at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.selection.pool_fetch import fetch_pool_embeddings


class _CountOnlyOS:
    def __init__(self, count: int) -> None:
        self._count = count
        self.search_calls = 0

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return {'count': self._count}

    async def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        self.search_calls += 1
        raise AssertionError('must not scroll a pool already known to exceed cap')


@pytest.mark.asyncio
async def test_pool_over_cap_is_truncated_without_scrolling() -> None:
    client = _CountOnlyOS(count=5000)
    ids, embeddings, truncated = await fetch_pool_embeddings(
        client, 'op_items', {'match_all': {}}, cap=100
    )
    assert truncated is True
    assert ids == []
    assert embeddings.shape == (0, 0)
    assert client.search_calls == 0


class _ScrollOS(_CountOnlyOS):
    def __init__(self, count: int, docs: dict[str, list[float]]) -> None:
        super().__init__(count)
        self._docs = docs

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        field = body['_source'][0]
        hits = [{'_id': cid, '_source': {field: emb}} for cid, emb in self._docs.items()]
        return {'_scroll_id': 'sid-1', 'hits': {'hits': hits}}

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}


@pytest.mark.asyncio
async def test_pool_under_cap_scrolls_normally() -> None:
    docs = {'a': [1.0, 0.0], 'b': [0.0, 1.0]}
    client = _ScrollOS(count=2, docs=docs)
    ids, embeddings, truncated = await fetch_pool_embeddings(
        client, 'op_items', {'match_all': {}}, cap=100
    )
    assert truncated is False
    assert set(ids) == set(docs)
    assert embeddings.shape == (2, 2)
