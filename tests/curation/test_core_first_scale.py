"""``order=core_first`` must not scan the whole cluster per request.

Every page is a native ``cluster_score`` sort (best-scored member first) of the cluster's members
in the serving partition: one page of documents, never the whole cluster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.paging_fakes import PagingOpenSearch
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config


if TYPE_CHECKING:
    import pytest

ITEMS = base_curation_config().items_index
UID = '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'
N = 4221
DIM = 8


class _CountingFake(PagingOpenSearch):
    """Counts the documents each call scans or returns."""

    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        super().__init__({ITEMS: docs})
        self.scanned = 0  # docs returned by search (scroll or page)
        self.search_calls = 0

    async def search(self, *, index: str, body: dict[str, Any], scroll: str | None = None, **kw):
        self.search_calls += 1
        if scroll is not None or not body.get('sort'):
            resp = await QueryFakeOpenSearch.search(
                self, index=index, body=body, scroll=scroll, **kw
            )
        else:
            resp = await super().search(index=index, body=body)
        self.scanned += len(resp['hits']['hits'])
        return resp


def _docs(n: int = N) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(0)
    docs: dict[str, dict[str, Any]] = {}
    for i in range(n):
        v = np.zeros(DIM, dtype=np.float32)
        v[0] = 1.0
        v[1:] = rng.normal(0, 0.3 if i % 5 else 1.5, DIM - 1)
        v /= np.linalg.norm(v)
        cid = f'c{i:05d}'
        docs[cid] = {
            'crop_id': cid,
            'cluster_uid': UID,
            'cluster_score': float(v[0]),
            'pe_embedding': v.tolist(),
        }
    docs['other'] = {
        'crop_id': 'other',
        'cluster_uid': 'someone-else',
        'cluster_score': 2.0,
        'pe_embedding': [1.0] + [0.0] * 7,
    }
    return docs


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router
    from src.services.curation.clustering import outliers

    outliers._CACHE.clear()
    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _get(client: TestClient, cursor: str | None, size: int = 60) -> dict[str, Any]:
    params: dict[str, Any] = {'cluster_uid': UID, 'order': 'core_first', 'limit': size}
    if cursor:
        params['cursor'] = cursor
    r = client.get('/curation/projects/default/crops', params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_core_first_pages_best_scored_members_without_a_cluster_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = _docs()
    fake = _CountingFake(docs)
    client = _client(fake, monkeypatch)

    p1 = _get(client, None)
    assert fake.scanned <= 61  # one page and its look-ahead row, not the 4k-member cluster
    assert fake.search_calls == 2  # hidden counts, the ordered page
    assert p1['total'] == N

    ids = [i for i, d in docs.items() if d['cluster_uid'] == UID]
    want = sorted(ids, key=lambda i: (-docs[i]['cluster_score'], i))
    got: list[str] = [c['crop_id'] for c in p1['crops']]
    cursor = p1['next_cursor']
    for _ in (2, 3):
        page = _get(client, cursor)
        got += [c['crop_id'] for c in page['crops']]
        cursor = page['next_cursor']
    assert got == want[:180]
    assert 'other' not in got
