"""The ``detector_disagreements`` review tab: unvalidated crops whose VLM class
differs from the class the detector itself gave (``detector_class_name``).

The disagreement test is a painless script (OpenSearch cannot compare two fields
otherwise); the in-memory fake evaluates it through a registered Python twin of
the same rule, so these tests pin the query shape and which seeded items it
selects, and the live check (plan step V) runs the real script.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation import query_fakes
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation import base_curation_config
from src.services.curation import review_queries, review_sorts
from src.services.curation.review_queries import DETECTOR_DISAGREES_SCRIPT
from src.services.curation.review_request import ReviewFilters, before_query, build_review_request


ITEMS = base_curation_config().items_index
URL = '/curation/projects/default/review/detector_disagreements'


def _disagrees(doc: dict[str, Any]) -> bool:
    detector, current = doc.get('detector_class_name'), doc.get('class_name')
    return bool(detector) and bool(current) and detector != current


@pytest.fixture(autouse=True)
def _script_twin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(query_fakes.SCRIPT_EVALUATORS, DETECTOR_DISAGREES_SCRIPT, _disagrees)


def _doc(crop_id: str, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_id': 1,
        'class_name': 'widget',
        'class_source': 'vlm',
        'class_validated': False,
        'detector_class_name': 'gadget',
        'detector_confidence': 0.7,
        'vlm_confidence': 'medium',
        'confidence': 0.7,
        **kw,
    }


def _docs() -> dict[str, dict[str, Any]]:
    docs = [
        _doc('disagree_med'),
        _doc('disagree_low', vlm_confidence='low', detector_confidence=0.9, confidence=0.9),
        _doc('disagree_high', vlm_confidence='high'),
        _doc('disagree_unrated', vlm_confidence=None),
        _doc('reclassified', class_source='vlm_reclassified'),
        _doc('agree', detector_class_name='widget'),
        _doc('validated', class_validated=True, class_source='human'),
        _doc('classifier_label', class_source='item_model'),
        _doc('no_detector_answer', detector_class_name=None),
        _doc('unmatched', class_name=None, class_id=None, class_source='vlm_unmatched'),
        _doc('dismissed', review_dismissed_at='2026-10-01T00:00:00+00:00'),
        _doc('excluded', class_excluded=True),
        _doc('holdout', test_holdout=True),
    ]
    return {d['crop_id']: d for d in docs}


@pytest.fixture
def fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch({ITEMS: _docs()})


@pytest.fixture
def client(fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from unittest.mock import AsyncMock

    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_only_unvalidated_vlm_disagreements_are_listed(client: TestClient) -> None:
    body = client.get(URL, params={'page_size': 100}).json()
    ids = {item['crop_id'] for item in body['items']}
    assert ids == {
        'disagree_med',
        'disagree_low',
        'disagree_high',
        'disagree_unrated',
        'reclassified',
    }
    assert body['total'] == 5  # the tab count is the listed set, nothing hidden
    assert {item['reason'] for item in body['items']} == {
        "VLM's class differs from the detector's class"
    }


@pytest.mark.asyncio
async def test_default_sort_is_vlm_then_detector_confidence() -> None:
    clause, applied, reason = await review_sorts.build_sort(None, tab='detector_disagreements')
    assert applied == 'detector_disagreement_default'
    assert reason is None
    assert clause == [
        review_sorts.VLM_CONFIDENCE_RANK_SORT,
        {'confidence': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
    ]


def test_tab_is_in_the_served_catalog_with_the_common_filters() -> None:
    tabs = {t['id']: t for t in review_queries.review_tab_catalog()}
    tab = tabs['detector_disagreements']
    assert tab['label'] == 'Detector disagreements'
    assert 'detector' in tab['description'].lower()
    assert set(tab['filters']) == set(review_queries.COMMON_FILTERS)


def test_the_unknown_tab_error_names_the_new_tab(client: TestClient) -> None:
    r = client.get('/curation/projects/default/review/not_a_tab')
    assert r.status_code == 400
    assert 'detector_disagreements' in r.text


def test_the_query_never_selects_a_validated_or_excluded_item() -> None:
    must, must_not, _ = review_queries.build_tab_query(
        'detector_disagreements', include_test=False, text=None, max_rank=None
    )
    assert {'term': {'class_validated': True}} in must_not
    assert {'term': {'class_excluded': True}} in must_not
    assert {'exists': {'field': 'detector_class_name'}} in must


def test_queue_lists_hard_cases_first(client: TestClient) -> None:
    items = client.get(URL, params={'page_size': 100}).json()['items']
    confidences = [item['vlm_confidence'] for item in items]
    assert confidences[0] is None
    assert confidences[1] == 'low'
    assert sorted(confidences[2:4]) == ['medium', 'medium']
    assert confidences[4] == 'high'


@pytest.mark.asyncio
async def test_locate_ranks_each_item_by_the_same_order(fake: QueryFakeOpenSearch) -> None:
    req = await build_review_request('detector_disagreements', ReviewFilters(), None, fake)
    resp = await fake.search(index=ITEMS, body={'query': req.query, 'sort': req.sort, 'size': 100})
    served = [hit['_id'] for hit in resp['hits']['hits']]
    assert len(served) == 5
    for position, crop_id in enumerate(served):
        source = {**fake.docs(ITEMS)[crop_id], 'crop_id': crop_id}
        before = {'bool': {'filter': [req.query, before_query(req.sort, source)]}}
        counted = await fake.count(index=ITEMS, body={'query': before})
        assert counted['count'] == position, (crop_id, served)
