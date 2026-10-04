"""``order=core_first`` must not scan the whole cluster per request.

The first request for a cluster without stored geometry computes live
and lazily writes the distances back; every later page is a native
``cluster_distance`` sort that touches one page of documents.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch, matches
from src.config.curation import base_curation_config
from src.services.curation import crop_orders
from src.services.curation.cluster_ids import CORE_SIMILARITY_MIN


ITEMS = base_curation_config().items_index
N = 4221
DIM = 8


class _CountingFake(QueryFakeOpenSearch):
    """Adds numeric multi-key sort + ``from`` + guarded script updates, and
    counts the documents each call scans or returns."""

    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        super().__init__({ITEMS: docs})
        self.scanned = 0  # docs returned by search (scroll or page)
        self.search_calls = 0

    async def search(self, *, index: str, body: dict[str, Any], scroll: str | None = None, **kw):  # type: ignore[no-untyped-def]
        self.search_calls += 1
        sort = body.get('sort')
        if scroll is not None or not sort:
            resp = await super().search(index=index, body=body, scroll=scroll, **kw)
            self.scanned += len(resp['hits']['hits'])
            return resp
        pool = [(i, d) for i, d in self.docs(index).items() if matches(d, body.get('query'))]
        for spec in reversed(sort):
            ((field, opts),) = spec.items()

            def _key(kv: tuple[str, dict[str, Any]], f: str = field) -> Any:
                return kv[1].get(f, kv[0])

            pool.sort(key=_key, reverse=opts['order'] == 'desc')
        start = body.get('from', 0)
        page = pool[start : start + body.get('size', 10)]
        self.scanned += len(page)
        return {
            'hits': {
                'total': {'value': len(pool)},
                'hits': [{'_id': i, '_source': copy.deepcopy(d)} for i, d in page],
            }
        }

    async def bulk(self, *, body: list[dict[str, Any]], **_kw: Any) -> dict[str, Any]:
        for action, payload in zip(body[::2], body[1::2], strict=True):
            doc = self.docs(action['update']['_index'])[action['update']['_id']]
            params = payload['script']['params']
            if doc.get('cluster_id') == params['cid']:
                doc.update(params['fields'])
        return {'errors': False, 'items': []}


def _docs(n: int = N) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(0)
    docs: dict[str, dict[str, Any]] = {}
    for i in range(n):
        v = np.zeros(DIM, dtype=np.float32)
        v[0] = 1.0
        v[1:] = rng.normal(0, 0.3 if i % 5 else 1.5, DIM - 1)
        v /= np.linalg.norm(v)
        cid = f'c{i:05d}'
        docs[cid] = {'crop_id': cid, 'cluster_id': 5, 'pe_embedding': v.tolist()}
    docs['other'] = {'crop_id': 'other', 'cluster_id': 6, 'pe_embedding': [1.0] + [0.0] * 7}
    return docs


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router
    from src.services.curation.clustering import outliers

    outliers._CACHE.clear()
    crop_orders._BACKFILLING.clear()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _get(client: TestClient, page: int, size: int = 60) -> dict[str, Any]:
    r = client.get(
        '/curation/projects/default/crops',
        params={'cluster_id': 5, 'order': 'core_first', 'page': page, 'page_size': size},
    )
    assert r.status_code == 200, r.text
    return r.json()


async def _drain() -> None:
    while crop_orders._TASKS:
        await asyncio.gather(*list(crop_orders._TASKS))


def test_stored_distances_serve_pages_without_a_cluster_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = _docs()
    # Pre-measured cluster (as the geometry pass leaves it).
    from src.services.curation.clustering.cluster_geometry import centroid_distances, unit_centroid

    ids = [i for i, d in docs.items() if d['cluster_id'] == 5]
    x = np.asarray([docs[i]['pe_embedding'] for i in ids], dtype=np.float32)
    dist = centroid_distances(x, unit_centroid(x))
    for i, dv in zip(ids, dist.tolist(), strict=True):
        docs[i]['cluster_distance'] = dv
        docs[i]['cluster_distance_cluster_id'] = 5
    fake = _CountingFake(docs)
    client = _client(fake, monkeypatch)

    p1 = _get(client, 1)
    # Bounded work: a page of docs, not the 4k-member cluster.
    assert fake.scanned <= 120  # router page + ordered page
    assert fake.search_calls == 2
    assert p1['total'] == N

    want = sorted(ids, key=lambda i: (docs[i]['cluster_distance'], i))
    got: list[str] = [c['crop_id'] for c in p1['crops']]
    for page in (2, 3):
        got += [c['crop_id'] for c in _get(client, page)['crops']]
    assert got == want[:180]
    flags = [c['cluster_is_core'] for c in p1['crops']]
    assert all(flags[: flags.index(False)] if False in flags else flags)


async def _page(fake: Any, page: int, size: int = 60) -> dict[str, Any]:
    from src.services.curation.item_filter import ItemFilter

    async def fetch(os_: Any, ids: list[str]) -> list[dict[str, Any]]:
        from src.services.curation.wire import serialize_item

        resp = await os_.mget(index=ITEMS, body={'ids': ids})
        return [serialize_item(d['_source'], d['_id']) for d in resp['docs']]

    clause = {'bool': {'filter': [{'term': {'cluster_id': 5}}]}}
    out = await crop_orders.ordered_crops_page(
        fake,
        index=ITEMS,
        order='core_first',
        query_clause=clause,
        cluster_id=5,
        item_filter=ItemFilter(),
        page=page,
        page_size=size,
        k=None,
        n_pool=len(fake.docs(ITEMS)),
        fetch_items=fetch,
    )
    assert out is not None
    return out


@pytest.mark.asyncio
async def test_unmeasured_cluster_computes_once_then_backfills() -> None:
    from src.services.curation.clustering import outliers

    outliers._CACHE.clear()
    crop_orders._BACKFILLING.clear()
    fake = _CountingFake(_docs(600))
    docs = fake.docs(ITEMS)

    first = await _page(fake, 1)
    d = [c['cluster_distance'] for c in first['crops']]
    assert d == sorted(d)
    assert all(
        c['cluster_is_core'] == (c['cluster_similarity'] >= CORE_SIMILARITY_MIN)
        for c in first['crops']
    )
    await _drain()
    measured = [
        i
        for i, x in docs.items()
        if x.get('cluster_distance_cluster_id') == 5 and 'cluster_distance' in x
    ]
    assert len(measured) == 600
    assert 'cluster_distance' not in docs['other']

    fake.scanned = 0
    second = await _page(fake, 1)
    assert fake.scanned <= 60
    assert [c['crop_id'] for c in second['crops']] == [c['crop_id'] for c in first['crops']]
    page2 = await _page(fake, 2)
    assert page2['crops'][0]['crop_id'] not in {c['crop_id'] for c in second['crops']}


@pytest.mark.asyncio
async def test_member_moved_in_falls_back_to_live_order() -> None:
    from src.services.curation.clustering import outliers

    outliers._CACHE.clear()
    crop_orders._BACKFILLING.clear()
    fake = _CountingFake(_docs(200))
    await _page(fake, 1)
    await _drain()
    fake.docs(ITEMS)['other']['cluster_id'] = 5  # unmeasured newcomer
    out = await _page(fake, 1, 500)
    assert out['total'] == 201
    assert 'other' in {c['crop_id'] for c in out['crops']}
    d = [c['cluster_distance'] for c in out['crops']]
    assert d == sorted(d)
