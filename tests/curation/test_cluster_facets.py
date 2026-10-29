"""``GET {P}/clusters/{cluster_id}/facets``: every count is what the matching filtered
``GET /crops`` returns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import TypeAdapter

from curation import query_fakes
from curation._cluster_cards_helpers import curation_client
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.schemas.curation.clusters import ClusterFacets
from src.services.curation import cluster_cards, review_tab_clauses


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


ITEMS = base_curation_config().items_index
URL = '/curation/projects/default'


def _disagrees(doc: dict[str, Any]) -> bool:
    detector, current = doc.get('detector_class_name'), doc.get('class_name')
    return bool(detector) and bool(current) and detector != current


def _item(crop_id: str, cluster_id: int = 1, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'cluster_id': cluster_id,
        'class_id': 1,
        'class_name': 'car',
        'class_source': 'vlm',
        'class_validated': False,
        'detector_class_name': 'car',
        'detector_confidence': 0.85,
        'vlm_confidence': 'high',
        **kw,
    }


def _docs() -> list[dict[str, Any]]:
    return [
        _item('a01', detector_confidence=0.05, class_validated=True, class_source='human'),
        _item('a02', detector_confidence=0.1),  # exactly on a bin edge: belongs to [0.1, 0.2)
        _item('a03', detector_confidence=0.5, detector_class_name='bus', vlm_confidence='low'),
        _item('a04', detector_confidence=0.55, detector_class_name='bus'),
        _item('a05', detector_confidence=1.0, class_validated=True, class_source='human'),
        _item('a06', detector_confidence=0.95, cluster_distance=0.5),
        _item('a07', detector_class_name=None, detector_confidence=None),
        _item('a08', class_name=None, class_id=None, class_source='vlm_unmatched'),
        _item('hold', test_holdout=True, detector_class_name='holdout_only'),
        _item('excl', class_excluded=True, detector_class_name='excluded_only'),
        _item('other', cluster_id=2, detector_class_name='other_cluster_only'),
    ]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> QueryFakeOpenSearch:
    monkeypatch.setitem(
        query_fakes.SCRIPT_EVALUATORS, review_tab_clauses.DETECTOR_DISAGREES_SCRIPT, _disagrees
    )
    return QueryFakeOpenSearch({ITEMS: {d['crop_id']: d for d in _docs()}})


@pytest.fixture
def client(fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return curation_client(fake, monkeypatch)


def _facets(client: TestClient, cluster_id: int = 1) -> dict[str, Any]:
    resp = client.get(f'{URL}/clusters/{cluster_id}/facets')
    assert resp.status_code == 200, resp.text
    return resp.json()


def _crops_total(client: TestClient, **params: Any) -> int:
    resp = client.get(f'{URL}/crops', params={'cluster_id': 1, 'page_size': 1, **params})
    assert resp.status_code == 200, resp.text
    return resp.json()['total']


def test_the_response_has_the_documented_shape(client: TestClient) -> None:
    body = _facets(client)
    TypeAdapter(ClusterFacets).validate_python(body, strict=True)
    assert body['cluster_id'] == 1
    # holdout and excluded members are not members for a reviewer
    assert body['total'] == 8
    assert _crops_total(client) == body['total']
    assert len(body['detector_confidence']) == 10
    assert set(body['review_tabs']) == set(cluster_cards.CLUSTER_REVIEW_TABS)


def test_other_clusters_holdout_and_excluded_values_never_appear(client: TestClient) -> None:
    values = {f['value'] for f in _facets(client)['detector_classes']}
    assert values == {'car', 'bus'}


def test_detector_class_counts_equal_the_filtered_crops(client: TestClient) -> None:
    facet = _facets(client)['detector_classes']
    assert {f['value']: f['count'] for f in facet} == {'car': 5, 'bus': 2}
    for entry in facet:
        assert entry['count'] == _crops_total(client, detector_class_name=entry['value'])


def test_class_source_counts_equal_the_filtered_crops(client: TestClient) -> None:
    facet = _facets(client)['class_sources']
    assert {f['value']: f['count'] for f in facet} == {'vlm': 5, 'human': 2, 'vlm_unmatched': 1}
    for entry in facet:
        assert entry['count'] == _crops_total(client, class_source=entry['value'])


def test_validation_counts_equal_the_filtered_crops(client: TestClient) -> None:
    validation = _facets(client)['validation']
    assert validation == {'validated': 2, 'unvalidated': 6}
    assert validation['validated'] == _crops_total(client, label_validated=True)
    assert validation['unvalidated'] == _crops_total(client, label_validated=False)


def test_review_tab_counts_equal_the_filtered_crops(client: TestClient) -> None:
    tabs = _facets(client)['review_tabs']
    for tab, count in tabs.items():
        assert count == _crops_total(client, review_tab=tab), tab
    assert tabs['outliers'] == 1
    assert tabs['detector_disagreements'] == 2
    assert tabs['all'] >= tabs['outliers']


def test_confidence_histogram_bins_are_the_filtered_crops_and_sum_to_the_total(
    client: TestClient,
) -> None:
    body = _facets(client)
    bins = body['detector_confidence']
    assert [b['min'] for b in bins] == [round(i / 10, 1) for i in range(10)]
    assert bins[-1]['max'] is None  # the last bin has no upper bound, so 1.0 is in it
    counts = [b['count'] for b in bins]
    assert counts[0] == 1  # 0.05
    assert counts[1] == 1  # 0.1 is the lower edge of the second bin
    assert counts[5] == 2  # 0.5, 0.55
    assert counts[9] == 2  # 0.95, 1.0
    assert sum(counts) + body['detector_confidence_missing'] == body['total']
    assert body['detector_confidence_missing'] == 1
    for b in bins:
        params: dict[str, Any] = {'detector_conf_min': b['min']}
        if b['max'] is not None:
            params['detector_conf_max'] = b['max']
        assert b['count'] == _crops_total(client, **params), b


def test_the_class_lists_are_capped_at_twenty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        query_fakes.SCRIPT_EVALUATORS, review_tab_clauses.DETECTOR_DISAGREES_SCRIPT, _disagrees
    )
    docs = [_item(f'm{i:02d}', detector_class_name=f'cls{i:02d}') for i in range(25)]
    client = curation_client(
        QueryFakeOpenSearch({ITEMS: {d['crop_id']: d for d in docs}}), monkeypatch
    )
    assert len(_facets(client)['detector_classes']) == 20


def test_an_unknown_cluster_has_empty_facets(client: TestClient) -> None:
    body = _facets(client, 999)
    assert body['total'] == 0
    assert body['detector_classes'] == []
    assert all(b['count'] == 0 for b in body['detector_confidence'])
    assert body['validation'] == {'validated': 0, 'unvalidated': 0}
