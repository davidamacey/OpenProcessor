"""One failing PIT slice stops the others: no orphan tasks, no search after the failure."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.services.curation.clustering.pit_pages import iter_pages
from src.services.curation.clustering.residual_fetch import fetch_residual_embeddings_parallel


class _SliceClient:
    """Slice 0 fails on its second search; the other slices never run dry."""

    def __init__(self, fail_slice: int | None = 0) -> None:
        self.fail_slice = fail_slice
        self.searches = 0
        self.searches_at_failure: int | None = None
        self.per_slice: dict[int, int] = {}
        self.deleted: list[Any] = []

    async def create_pit(self, **_kw: Any) -> dict[str, str]:
        return {'pit_id': 'pit1'}

    async def delete_pit(self, **kw: Any) -> None:
        self.deleted.append(kw)

    async def count(self, **_kw: Any) -> dict[str, int]:
        return {'count': 0}

    async def search(self, *, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        slice_id = body['slice']['id']
        n = self.per_slice[slice_id] = self.per_slice.get(slice_id, 0) + 1
        self.searches += 1
        if slice_id == self.fail_slice and n == 2:
            self.searches_at_failure = self.searches
            raise RuntimeError('shard failure')
        doc = {'_id': f'{slice_id}-{n}', '_source': {'pe_embedding': [1.0, 0.0]}, 'sort': [n]}
        return {'hits': {'hits': [doc]}}


async def _no_orphans_and_no_more_searches(client: _SliceClient) -> None:
    after = client.searches
    await asyncio.sleep(0.05)
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert others == []
    assert client.searches == after


@pytest.mark.asyncio
async def test_iter_pages_cancels_the_other_slices_when_one_fails() -> None:
    client = _SliceClient()
    with pytest.raises(RuntimeError, match='shard failure'):
        async for _page in iter_pages(client, index='i', query={}, source=['x'], n_slices=3):  # type: ignore[arg-type]
            pass
    await _no_orphans_and_no_more_searches(client)
    assert client.deleted == [{'body': {'pit_id': ['pit1']}}]


@pytest.mark.asyncio
async def test_residual_fetch_cancels_the_other_slices_when_one_fails() -> None:
    client = _SliceClient()
    with pytest.raises(RuntimeError, match='shard failure'):
        await fetch_residual_embeddings_parallel(
            client, field='pe_embedding', n_slices=3, page_size=1
        )  # type: ignore[arg-type]
    await _no_orphans_and_no_more_searches(client)
    assert client.deleted == [{'body': {'pit_id': ['pit1']}}]


@pytest.mark.asyncio
async def test_a_clean_scan_yields_every_page_once() -> None:
    class Finite(_SliceClient):
        async def search(self, *, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
            slice_id = body['slice']['id']
            if self.per_slice.get(slice_id, 0) >= 2:
                return {'hits': {'hits': []}}
            return await super().search(body=body, **kw)

    client = Finite(fail_slice=None)
    ids = []
    async for page in iter_pages(client, index='i', query={}, source=['x'], n_slices=3):  # type: ignore[arg-type]
        ids.extend(h['_id'] for h in page)
    assert sorted(ids) == sorted(f'{s}-{n}' for s in range(3) for n in (1, 2))
