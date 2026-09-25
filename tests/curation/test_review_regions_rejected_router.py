"""``GET /review/regions`` end-to-end: a verifier-rejected candidate is now
reachable from the review queue (DQ-B2 follow-up).

Unit-level match logic lives in ``test_review_regions_tab.py``; this
exercises the real router (``region_status`` query param wiring, the
``filter_specs`` catalog on ``GET /review/tabs``, and the per-item
``reason``) against ``QueryFakeOpenSearch``, which actually evaluates the
query/sort instead of just recording the call.
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


def _docs() -> dict[str, dict[str, Any]]:
    return {
        'detected1': {
            'crop_id': 'detected1',
            F.bbox_norm: [0.1, 0.1, 0.2, 0.2],
            F.status: 'detected',
            F.score: 0.6,
        },
        'rejected1': {
            'crop_id': 'rejected1',
            F.status: 'verify_rejected',
            F.candidate_bbox_norm: [0.3, 0.6, 0.4, 0.65],
            F.candidate_score: 0.81,
            F.rejection_reason: 'region_visible_elsewhere',
        },
        'validated': {
            'crop_id': 'validated',
            F.bbox_norm: [0.1, 0.1, 0.2, 0.2],
            F.status: 'detected',
            F.validated: True,
        },
        'fp': {
            'crop_id': 'fp',
            F.bbox_norm: [0.1, 0.1, 0.2, 0.2],
            F.status: 'false_positive',
        },
    }


def test_default_queue_includes_rejected_candidate_and_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), monkeypatch)
    r = client.get('/curation/review/regions')
    assert r.status_code == 200, r.text
    crop_ids = {i['crop_id'] for i in r.json()['items']}
    assert crop_ids == {'detected1', 'rejected1'}


def test_detected_only_filter_excludes_rejected_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), monkeypatch)
    r = client.get('/curation/review/regions', params={'region_status': 'detected'})
    assert r.status_code == 200, r.text
    assert [i['crop_id'] for i in r.json()['items']] == ['detected1']


def test_verify_rejected_only_filter_excludes_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), monkeypatch)
    r = client.get('/curation/review/regions', params={'region_status': 'verify_rejected'})
    assert r.status_code == 200, r.text
    assert [i['crop_id'] for i in r.json()['items']] == ['rejected1']


def test_human_validated_items_still_excluded_in_every_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), monkeypatch)
    for params in ({}, {'region_status': 'all'}, {'region_status': 'detected'}):
        r = client.get('/curation/review/regions', params=params)
        assert 'validated' not in {i['crop_id'] for i in r.json()['items']}, params


def test_false_positive_never_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), monkeypatch)
    for params in ({}, {'region_status': 'detected'}, {'region_status': 'verify_rejected'}):
        r = client.get('/curation/review/regions', params=params)
        assert 'fp' not in {i['crop_id'] for i in r.json()['items']}, params


def test_unknown_region_status_filter_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), monkeypatch)
    r = client.get('/curation/review/regions', params={'region_status': 'bogus'})
    assert r.status_code == 400
    assert 'region_status' in r.json()['detail']


def test_rejected_item_reason_mentions_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """R10: the reason is worded from the served rejection-reason
    vocabulary label, not the raw region_rejection_reason id verbatim."""
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), monkeypatch)
    r = client.get('/curation/review/regions')
    items = {i['crop_id']: i for i in r.json()['items']}
    assert items['rejected1']['reason'] == 'rejected: the detection is wrong (region is elsewhere)'
    assert 'region_visible_elsewhere' not in items['rejected1']['reason']
    assert items['detected1']['reason'] == 'region detected — needs human confirmation'


def test_rejected_items_sort_by_candidate_score_not_arbitrary_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both rejected items are missing `region_score` (tie on the first
    sort key); the second key (`region_candidate_score`) must still order
    them deterministically by score, not by insertion/shard order."""
    docs = {
        'rej_low': {
            'crop_id': 'rej_low',
            F.status: 'verify_rejected',
            F.candidate_bbox_norm: [0.3, 0.6, 0.4, 0.65],
            F.candidate_score: 0.2,
        },
        'rej_high': {
            'crop_id': 'rej_high',
            F.status: 'verify_rejected',
            F.candidate_bbox_norm: [0.3, 0.6, 0.4, 0.65],
            F.candidate_score: 0.9,
        },
    }
    from src.services.curation import review_sorts

    registry = review_sorts.get_review_sorts()
    clause = registry['region_score'].clause
    assert clause[0][F.score]['order'] == 'desc'
    assert clause[1][F.candidate_score]['order'] == 'desc'

    # The fake's search() only sorts on the first field, so assert the
    # ordering semantics directly against the clause + locate's own
    # generic before_query rather than relying on the fake's sort.
    client = _client(QueryFakeOpenSearch({ITEMS: docs}), monkeypatch)
    lo = client.get('/curation/review/regions/locate', params={'crop_id': 'rej_low'}).json()
    hi = client.get('/curation/review/regions/locate', params={'crop_id': 'rej_high'}).json()
    assert lo['in_queue']
    assert hi['in_queue']
    # Higher candidate score sorts strictly before ('rank' is the count of
    # items sorting before it) the lower one.
    assert hi['rank'] < lo['rank']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
