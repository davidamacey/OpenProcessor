"""``GET /detections/summary``: labels x embedding state, scoped by the shared filter."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config
from src.routers.curation import _common


P = f'{_common.config.api_prefix}/projects/default'


def _doc(
    crop_id: str, name: str | None, state: str | None, *, vector: bool = False
) -> dict[str, Any]:
    doc: dict[str, Any] = {'crop_id': crop_id, 'confidence': 0.6}
    if name:
        doc['proposal_name'] = name
    if state:
        doc['embedding_state'] = state
    if vector:
        doc['pe_embedding'] = [1.0]
    return doc


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    docs = [
        _doc('a', 'person', 'embedded', vector=True),
        _doc('b', 'person', 'not_selected'),
        _doc('c', 'person', 'not_selected'),
        _doc('d', 'car', 'embedded', vector=True),
        _doc('e', 'car', 'failed'),
        _doc('f', 'dog', None),
        _doc('g', None, 'deferred'),
    ]
    fake = QueryFakeOpenSearch({get_curation_config().items_index: {d['crop_id']: d for d in docs}})
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', {'default'})
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def test_counts_each_label_with_its_embedding_breakdown(client: TestClient) -> None:
    body = client.get(f'{P}/detections/summary').json()
    assert body['total'] == 7
    assert body['embedding'] == {
        'embedded': 2,
        'not_embedded': 5,
        'by_state': {'embedded': 2, 'not_selected': 2, 'failed': 1, 'deferred': 1, 'unknown': 1},
    }
    by = {label['name']: label for label in body['by_label']}
    assert by['person']['count'] == 3
    assert by['person']['embedding'] == {
        'embedded': 1,
        'not_embedded': 2,
        'by_state': {'embedded': 1, 'not_selected': 2},
    }
    assert by['car']['embedding']['by_state'] == {'embedded': 1, 'failed': 1}
    assert by['(no label)']['count'] == 1
    assert body['labels_truncated'] is False


def test_the_shared_filter_scopes_the_summary(client: TestClient) -> None:
    body = client.get(f'{P}/detections/summary', params={'class_name': 'person'}).json()
    assert body['total'] == 3
    assert [label['name'] for label in body['by_label']] == ['person']


def test_the_suggested_reprocess_is_postable_and_selects_the_missing(client: TestClient) -> None:
    body = client.get(f'{P}/detections/summary', params={'class_name': 'person'}).json()
    suggestion = body['suggested_reprocess']
    assert suggestion['scopes'] == ['embed']
    assert suggestion['embed']['only_missing'] is True
    assert suggestion['dry_run'] is True
    flt = suggestion['targets']['filter']
    assert flt['class_names'] == ['person']
    assert flt['embedding_state'] == ['not_selected', 'deferred', 'failed']
    posted = client.post(f'{P}/reprocess', json=suggestion)
    assert posted.status_code == 200, posted.text
    assert posted.json()['dry_run'] is True


def test_no_suggestion_when_everything_is_embedded(client: TestClient) -> None:
    body = client.get(f'{P}/detections/summary', params={'embedding_state': 'embedded'}).json()
    assert body['embedding']['not_embedded'] == 0
    assert body['suggested_reprocess'] is None


def test_a_malformed_band_is_a_400(client: TestClient) -> None:
    r = client.get(f'{P}/detections/summary', params={'conf_min': '0.9', 'conf_max': '0.1'})
    assert r.status_code == 400
