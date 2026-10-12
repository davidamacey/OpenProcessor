"""Request shapes of the benchmark's plain-REST OpenSearch adapter (no client library)."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from scripts.bench.vectordb.adapters.opensearch import INDEX, OpenSearchAdapter
from scripts.bench.vectordb.adapters.os_rest import Rest
from scripts.bench.vectordb.vdb_lib import FilterSpec


def _adapter(arm: dict, seen: list[httpx.Request], reply: dict | None = None) -> OpenSearchAdapter:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=reply if reply is not None else {})

    ad = OpenSearchAdapter(arm, '127.0.0.1:1')
    ad.client = Rest('127.0.0.1:1', transport=httpx.MockTransport(handler))
    return ad


def test_search_posts_knn_with_filter_and_method_parameters() -> None:
    seen: list[httpx.Request] = []
    hits = {'hits': {'hits': [{'_id': '7'}, {'_id': '3'}]}}
    ad = _adapter({}, seen, hits)
    flt = FilterSpec('f', eq=(('project', 2),), is_false=('test_holdout',), sel_lt=50)
    got = ad.search(np.ones(4, np.float32), 5, flt, {'ef_search': 64})
    assert got == [7, 3]
    req = seen[0]
    assert req.method == 'POST'
    assert req.url.path == f'/{INDEX}/_search'
    assert req.url.params['request_cache'] == 'false'
    knn = json.loads(req.content)['query']['knn']['vec']
    assert knn['k'] == 5
    assert knn['method_parameters'] == {'ef_search': 64}
    assert knn['filter']['bool']['filter'] == [
        {'term': {'project': 2}},
        {'term': {'test_holdout': False}},
        {'range': {'sel': {'lt': 50}}},
    ]


def test_lucene_search_never_asks_for_ef_below_k() -> None:
    seen: list[httpx.Request] = []
    ad = _adapter({'knn_engine': 'lucene'}, seen, {'hits': {'hits': []}})
    ad.search(np.ones(4, np.float32), 100, FilterSpec('none'), {'ef_search': 16})
    assert json.loads(seen[0].content)['query']['knn']['vec']['method_parameters'] == {
        'ef_search': 100
    }


def test_bulk_is_ndjson_and_retries_only_429() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert req.headers['content-type'] == 'application/x-ndjson'
        assert req.content.endswith(b'\n')
        if len(calls) == 1:
            return httpx.Response(200, json={'errors': True, 'items': [{'index': {'status': 429}}]})
        return httpx.Response(200, json={'errors': False, 'items': []})

    ad = OpenSearchAdapter({}, '127.0.0.1:1')
    ad.client = Rest('127.0.0.1:1', transport=httpx.MockTransport(handler))
    ad.send_bulk([{'update': {'_index': INDEX, '_id': '1'}}, {'doc': {'cluster_id': 3}}])
    assert len(calls) == 2
    assert calls[0].url.path == '/_bulk'

    def hard_error(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'errors': True, 'items': [{'index': {'status': 400}}]})

    ad.client = Rest('127.0.0.1:1', transport=httpx.MockTransport(hard_error))
    with pytest.raises(RuntimeError):
        ad.send_bulk('x\n')


def test_admin_calls_use_the_documented_paths() -> None:
    seen: list[httpx.Request] = []
    ad = _adapter({}, seen, {'count': 4})
    ad.client.refresh(INDEX)
    ad.client.forcemerge(INDEX)
    ad.client.warmup(INDEX)
    ad.delete_where(FilterSpec('p', eq=(('project', 0),)))
    assert ad.count() == 4
    got = [(r.method, r.url.path) for r in seen]
    assert got == [
        ('POST', f'/{INDEX}/_refresh'),
        ('POST', f'/{INDEX}/_forcemerge'),
        ('GET', f'/_plugins/_knn/warmup/{INDEX}'),
        ('POST', f'/{INDEX}/_delete_by_query'),
        ('GET', f'/{INDEX}/_count'),
    ]
    assert seen[1].url.params['max_num_segments'] == '1'
    assert seen[3].url.params['wait_for_completion'] == 'true'


def test_http_errors_raise() -> None:
    rest = Rest(
        '127.0.0.1:1', transport=httpx.MockTransport(lambda _req: httpx.Response(500, text='boom'))
    )
    with pytest.raises(RuntimeError):
        rest.refresh(INDEX)
