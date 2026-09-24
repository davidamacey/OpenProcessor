"""Review-queue additions: locate, shared filters, the new-class queue.

- ``GET /review/{tab}/locate?crop_id=`` answers where a crop sits in a
  queue (rank + page) by *counting* the items that sort before it under
  the same query and sort — so a deep link works at any queue depth.
  Every queue sort ends in a ``crop_id`` tiebreak so that rank is exact.
- ``/review/{tab}`` honours ``class_id``, ``source`` and the
  ``conf_min``/``conf_max`` band.
- ``new_class_proposals`` lists human-flagged and VLM-proposed new-class
  items; ``/review/new_class_proposals/summary`` aggregates the VLM's
  proposed names.
- A tab's own default sort beats a deployment-wide sort default.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config, get_region_fields


ITEMS = get_curation_config().items_index
F = get_region_fields()


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr(
        'src.services.curation.strategy_registry.resolve_effective_default',
        AsyncMock(return_value=None),
    )
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _region_docs() -> dict[str, dict[str, Any]]:
    scores = {'r01': 0.9, 'r02': 0.5, 'r03': 0.9, 'r04': None, 'r05': 0.7, 'r06': None, 'r07': 0.5}
    docs = {}
    for cid, score in scores.items():
        doc: dict[str, Any] = {'crop_id': cid, F.bbox_norm: [0.1, 0.1, 0.2, 0.2]}
        if score is not None:
            doc[F.score] = score
        docs[cid] = doc
    docs['done'] = {'crop_id': 'done', F.bbox_norm: [0.1, 0.1, 0.2, 0.2], F.validated: True}
    return docs


def _expected_regions_order(docs: dict[str, dict[str, Any]]) -> list[str]:
    queue = [d for d in docs.values() if not d.get(F.validated)]
    queue.sort(key=lambda d: (d.get(F.score) is None, -(d.get(F.score) or 0.0), d['crop_id']))
    return [d['crop_id'] for d in queue]


def test_locate_rank_matches_queue_order(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = _region_docs()
    client = _client(QueryFakeOpenSearch({ITEMS: docs}), monkeypatch)
    order = _expected_regions_order(docs)
    for rank, cid in enumerate(order):
        r = client.get('/curation/review/regions/locate', params={'crop_id': cid, 'page_size': 3})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body['in_queue'] is True, body
        assert body['rank'] == rank, (cid, body)
        assert body['page'] == rank // 3 + 1
        assert body['sort_applied'] == 'region_score'


def test_locate_reports_not_in_queue_and_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _region_docs()}), monkeypatch)
    body = client.get('/curation/review/regions/locate', params={'crop_id': 'done'}).json()
    assert (body['in_queue'], body['rank'], body['page'], body['reason']) == (
        False,
        None,
        None,
        'filtered_out',
    )
    body = client.get('/curation/review/regions/locate', params={'crop_id': 'nope'}).json()
    assert (body['in_queue'], body['reason']) == (False, 'not_found')


def test_queue_sort_ends_in_crop_id_tiebreak(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {}})
    fake.search = AsyncMock(return_value={'hits': {'total': {'value': 0}, 'hits': []}})  # type: ignore[method-assign]
    _client(fake, monkeypatch).get('/curation/review/all')
    assert fake.search.call_args.kwargs['body']['sort'][-1] == {'crop_id': {'order': 'asc'}}


def test_review_filters_class_source_and_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    base = {'class_source': 'vlm_unmatched'}
    docs = {
        'a': {**base, 'crop_id': 'a', 'class_id': 1, 'hdd_source': 's1', 'confidence': 0.3},
        'b': {**base, 'crop_id': 'b', 'class_id': 2, 'hdd_source': 's1', 'confidence': 0.3},
        'c': {**base, 'crop_id': 'c', 'class_id': 1, 'hdd_source': 's2', 'confidence': 0.3},
        'd': {**base, 'crop_id': 'd', 'class_id': 1, 'hdd_source': 's1', 'confidence': 0.9},
    }
    client = _client(QueryFakeOpenSearch({ITEMS: docs}), monkeypatch)
    r = client.get(
        '/curation/review/mismatches',
        params={'class_id': 1, 'source': 's1', 'conf_min': 0.1, 'conf_max': 0.5},
    )
    assert r.status_code == 200, r.text
    assert [i['crop_id'] for i in r.json()['items']] == ['a']


def test_new_class_proposals_tab_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        'flag': {'crop_id': 'flag', 'needs_new_class': True, 'needs_new_class_note': 'trailer'},
        'p1': {
            'crop_id': 'p1',
            'class_source': 'vlm_new_class_pending',
            'vlm_proposed_class': 'trailer',
        },
        'p2': {
            'crop_id': 'p2',
            'class_source': 'vlm_new_class_pending',
            'vlm_proposed_class': 'trailer',
        },
        'p3': {
            'crop_id': 'p3',
            'class_source': 'vlm_new_class_pending',
            'vlm_proposed_class': 'kayak',
        },
        'plain': {'crop_id': 'plain', 'class_source': 'vlm'},
    }
    client = _client(QueryFakeOpenSearch({ITEMS: docs}), monkeypatch)
    r = client.get('/curation/review/new_class_proposals')
    assert r.status_code == 200, r.text
    assert sorted(i['crop_id'] for i in r.json()['items']) == ['flag', 'p1', 'p2', 'p3']

    r = client.get('/curation/review/new_class_proposals/summary')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['total_pending'] == 3
    terms = {t['label']: t for t in body['top_terms']}
    assert terms['trailer']['count'] == 2
    assert sorted(terms['trailer']['sample_crop_ids']) == ['p1', 'p2']
    assert terms['kayak']['sample_crop_ids'] == ['p3']


@pytest.mark.asyncio
async def test_tab_default_beats_deployment_sort_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import review_sorts

    monkeypatch.setattr(
        'src.services.curation.strategy_registry.resolve_effective_default',
        AsyncMock(return_value='uncertainty_entropy'),
    )
    _clause, applied, _ = await review_sorts.build_sort(None, tab='regions', opensearch=object())
    assert applied == 'region_score'
    # A tab with no default of its own takes the deployment default.
    _clause, applied, _ = await review_sorts.build_sort(
        None, tab='new_class_proposals', opensearch=object()
    )
    assert applied == 'uncertainty_entropy'
