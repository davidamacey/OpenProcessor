"""The detector filters of the shared item filter (``detector_class_name``,
``detector_conf_min`` / ``detector_conf_max``, ``review_tab``) and the
``detector_confidence`` sort of ``GET /crops``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation import query_fakes
from curation._cluster_cards_helpers import curation_client
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation import review_queries, review_tab_clauses
from src.services.curation.item_filter import ItemFilter, item_filter_clauses


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


ITEMS = base_curation_config().items_index
URL = '/curation/projects/default'


def test_detector_class_names_match_the_stored_spellings() -> None:
    (clause,) = item_filter_clauses(ItemFilter(detector_class_names=['Traffic Light', 'car']))
    terms = {
        (field, spec['value'])
        for c in clause['bool']['should']
        for field, spec in c['term'].items()
    }
    assert {f for f, _ in terms} == {'detector_class_name'}
    assert {v for _, v in terms} >= {'traffic_light', 'traffic light', 'traffic-light', 'car'}


def test_detector_confidence_band_is_inclusive_below_and_exclusive_above() -> None:
    (clause,) = item_filter_clauses(ItemFilter(detector_conf_min=0.2, detector_conf_max=0.3))
    assert clause == {'range': {'detector_confidence': {'gte': 0.2, 'lt': 0.3}}}
    (low,) = item_filter_clauses(ItemFilter(detector_conf_min=0.9))
    assert low == {'range': {'detector_confidence': {'gte': 0.9}}}


@pytest.mark.parametrize(('lo', 'hi'), [(0.5, 0.5), (0.6, 0.5)])
def test_an_empty_detector_band_is_refused(lo: float, hi: float) -> None:
    with pytest.raises(ValueError, match='detector_conf_min'):
        item_filter_clauses(ItemFilter(detector_conf_min=lo, detector_conf_max=hi))


def test_review_tab_is_the_review_queues_own_clause() -> None:
    (clause,) = item_filter_clauses(ItemFilter(review_tab='outliers'))
    must, must_not, _reason = review_queries.build_tab_query(
        'outliers', include_test=False, text=None, max_rank=None
    )
    assert clause == {'bool': {'must': must, 'must_not': must_not}}


@pytest.mark.parametrize('tab', ['regions', 'nonsense'])
def test_review_tab_must_be_a_simple_tab(tab: str) -> None:
    with pytest.raises(ValueError, match='review_tab'):
        item_filter_clauses(ItemFilter(review_tab=tab))
    assert 'outliers' in review_tab_clauses.SIMPLE_TABS


# --- end to end through GET /crops and GET /review/{tab} --------------------------------


def _disagrees(doc: dict[str, Any]) -> bool:
    detector, current = doc.get('detector_class_name'), doc.get('class_name')
    return bool(detector) and bool(current) and detector != current


def _item(crop_id: str, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'cluster_id': 1,
        'class_id': 1,
        'class_name': 'car',
        'class_source': 'vlm',
        'class_validated': False,
        'detector_class_name': 'car',
        'detector_confidence': 0.8,
        'vlm_confidence': 'high',
        'updated_at': '2026-01-01T00:00:00+00:00',
        **kw,
    }


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> QueryFakeOpenSearch:
    monkeypatch.setitem(
        query_fakes.SCRIPT_EVALUATORS, review_tab_clauses.DETECTOR_DISAGREES_SCRIPT, _disagrees
    )
    docs = [
        _item('c1', detector_confidence=0.95),
        _item('c2', detector_confidence=0.5),
        _item('c3', detector_confidence=0.2, detector_class_name='bus'),
        _item('c4', detector_confidence=0.1, detector_class_name='traffic_light'),
        _item('c5', detector_class_name=None, detector_confidence=None),
        _item('c6', cluster_distance=0.6, detector_confidence=0.7),
        _item('c7', test_holdout=True, detector_confidence=0.7, cluster_distance=0.6),
    ]
    return QueryFakeOpenSearch({ITEMS: {d['crop_id']: d for d in docs}})


@pytest.fixture
def client(fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return curation_client(fake, monkeypatch)


def _ids(client: TestClient, **params: Any) -> list[str]:
    body = client.get(f'{URL}/crops', params={'page_size': 100, **params}).json()
    return [c['crop_id'] for c in body['crops']]


def test_crops_filter_by_detector_class(client: TestClient) -> None:
    assert sorted(_ids(client, detector_class_name='bus')) == ['c3']
    assert sorted(_ids(client, detector_class_name=['bus', 'Traffic Light'])) == ['c3', 'c4']


def test_crops_filter_by_detector_confidence_band(client: TestClient) -> None:
    # [0.5, 0.8): the upper bound is exclusive so adjacent histogram bins never overlap.
    assert sorted(_ids(client, detector_conf_min=0.5, detector_conf_max=0.8)) == ['c2', 'c6']
    assert sorted(_ids(client, detector_conf_min=0.5, detector_conf_max=0.7)) == ['c2']
    assert sorted(_ids(client, detector_conf_min=0.9)) == ['c1']
    assert 'c5' not in _ids(client, detector_conf_max=1.0)  # no detector confidence: no match


def test_crops_filter_by_review_tab_equals_the_review_route(client: TestClient) -> None:
    in_crops = sorted(_ids(client, review_tab='outliers'))
    in_review = sorted(
        i['crop_id']
        for i in client.get(f'{URL}/review/outliers', params={'page_size': 100}).json()['items']
    )
    assert in_crops == in_review == ['c6']


def test_review_tab_and_detector_class_combine(client: TestClient) -> None:
    both = _ids(client, review_tab='detector_disagreements', detector_class_name='bus')
    assert both == ['c3']
    assert _ids(client, review_tab='outliers', detector_class_name='bus') == []
    assert _ids(client, review_tab='outliers', detector_class_name='car') == ['c6']


def test_crops_sort_by_detector_confidence(
    client: TestClient, fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fake orders by string key, so the order itself is pinned on the clause the
    # route sends: missing values last in either direction, ties on crop_id.
    sent: list[dict[str, Any]] = []
    real = fake.search

    async def spy(**kw: Any) -> Any:
        sent.append(kw['body'])
        return await real(**kw)

    monkeypatch.setattr(fake, 'search', spy)
    for direction in ('asc', 'desc'):
        resp = client.get(f'{URL}/crops', params={'sort': f'detector_confidence:{direction}'})
        assert resp.status_code == 200, resp.text
        assert sent[-1]['sort'] == [
            {
                'detector_confidence': {
                    'order': direction,
                    'missing': '_last',
                    'unmapped_type': 'float',
                }
            },
            {'crop_id': {'order': 'asc'}},
        ]


def test_a_bad_band_or_tab_is_a_400(client: TestClient) -> None:
    band = {'detector_conf_min': 0.9, 'detector_conf_max': 0.1}
    assert client.get(f'{URL}/crops', params=band).status_code == 400
    assert client.get(f'{URL}/crops', params={'review_tab': 'regions'}).status_code == 400
