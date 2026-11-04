"""A scroll context is cleared on the error path too, not only after a clean loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.clients.opensearch.client import OpenSearchClient
from src.clients.opensearch.names import IndexName
from src.clients.opensearch.scroll import scroll_ids
from src.utils.upstream_errors import UpstreamError


class _Scrolling:
    """A scroll that serves ``pages`` and raises on the page numbered ``fail_on``."""

    def __init__(self, pages: int = 2, fail_on: int | None = None) -> None:
        self.pages, self.fail_on = pages, fail_on
        self.served = 0
        self.cleared: list[str] = []

    def _page(self) -> dict[str, Any]:
        if self.fail_on is not None and self.served + 1 == self.fail_on:
            raise RuntimeError('shard failure')
        self.served += 1
        hits = (
            [{'_id': f'd{self.served}', '_source': {'embedding': [0.0] * 4, 'detection_id': 'x'}}]
            if self.served <= self.pages
            else []
        )
        return {'_scroll_id': 's1', 'hits': {'hits': hits}}

    async def search(self, **_kw: Any) -> dict[str, Any]:
        return self._page()

    async def scroll(self, **_kw: Any) -> dict[str, Any]:
        return self._page()

    async def clear_scroll(self, scroll_id: str) -> None:
        self.cleared.append(scroll_id)


@pytest.mark.asyncio
async def test_scroll_ids_clears_the_scroll_when_page_two_raises() -> None:
    fake = _Scrolling(fail_on=2)
    with pytest.raises(RuntimeError, match='shard failure'):
        await scroll_ids(fake, index='i', query={'match_all': {}})
    assert fake.cleared == ['s1']


@pytest.mark.asyncio
async def test_scroll_ids_clears_once_on_a_clean_run() -> None:
    fake = _Scrolling()
    assert await scroll_ids(fake, index='i', query={'match_all': {}}) == ['d1', 'd2']
    assert fake.cleared == ['s1']


def _client(fake: _Scrolling) -> OpenSearchClient:
    c = OpenSearchClient(hosts=['http://localhost:9200'])
    c.client = fake  # type: ignore[assignment]
    c.client.indices = MagicMock()
    return c


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['get_all_embeddings', 'get_unclustered_embeddings'])
async def test_cluster_extraction_clears_the_scroll_on_error(method: str) -> None:
    fake = _Scrolling(fail_on=2)
    with pytest.raises(UpstreamError):
        await getattr(_client(fake), method)(IndexName.GLOBAL)
    assert fake.cleared == ['s1']


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['get_all_embeddings', 'get_unclustered_embeddings'])
async def test_cluster_extraction_clears_once_on_a_clean_run(method: str) -> None:
    fake = _Scrolling()
    emb, _ids = await getattr(_client(fake), method)(IndexName.GLOBAL)
    assert fake.cleared == ['s1']
    assert isinstance(emb, np.ndarray)
