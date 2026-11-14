"""The near-duplicate OpenSearch stand-in keeps the point-in-time semantics the page reader relies on."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from opensearchpy.exceptions import NotFoundError

from src.services.curation.clustering.pit_pages import iter_pages

from .fake_opensearch import FakeNearDupOpenSearch


def _docs(n: int) -> dict[str, dict[str, dict]]:
    return {'idx': {f'd{i:02d}': {'v': [float(i), 1.0], 'g': i % 2} for i in range(n)}}


@pytest.mark.asyncio
async def test_pages_are_a_snapshot_disjoint_across_slices_and_complete() -> None:
    fake = FakeNearDupOpenSearch(_docs(25))
    seen: list[str] = []
    async for hits in iter_pages(
        fake,  # type: ignore[arg-type]  # the stand-in is the client
        index='idx', query={'exists': {'field': 'v'}}, source=['g'], docvalue_fields=['v'],
        n_slices=3, page_size=4,
    ):  # fmt: skip
        for h in hits:
            seen.append(h['_id'])
            assert h['_source'] == {'g': int(h['_id'][1:]) % 2}
            raw = base64.b64decode(h['fields']['v'][0])
            assert np.frombuffer(raw, dtype='<f4').tolist() == [float(h['_id'][1:]), 1.0]
    assert sorted(seen) == sorted(fake.docs['idx'])  # every document once
    assert fake.pits == {}  # the reader freed its view


@pytest.mark.asyncio
async def test_a_view_ignores_later_writes_and_a_freed_one_is_refused() -> None:
    fake = FakeNearDupOpenSearch(_docs(3))
    pit = (await fake.create_pit(index='idx'))['pit_id']
    fake.docs['idx']['late'] = {'v': [9.0, 9.0], 'g': 0}
    body = {'size': 10, 'query': {'exists': {'field': 'v'}}, '_source': False, 'pit': {'id': pit}}
    got = await fake.search(body=body)
    assert [h['_id'] for h in got['hits']['hits']] == ['d00', 'd01', 'd02']
    assert 'sort' in got['hits']['hits'][0]
    await fake.delete_pit(body={'pit_id': [pit]})
    with pytest.raises(NotFoundError):
        await fake.search(body=body)
