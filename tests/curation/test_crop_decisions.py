"""Smaller backend-owned decisions and read-outs.

- ``POST /crops/{id}/vlm_dismiss``: the operator rejects the VLM's class
  suggestion; it stops being suggested (wire suggestion keys go null and
  a one-key confirm no longer applies it) until the VLM suggests
  something else.
- ``GET /crops/{id}/history``: the item's class history for an audit panel,
  with only the documented keys.
- ``GET /regions/suspected_false_positives`` defaults its threshold on the
  server and reports the default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config
from src.services.curation.wire import serialize_item


if TYPE_CHECKING:
    import pytest


ITEMS = get_curation_config().items_index


def _client(fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _vlm_doc(crop_id: str, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_id': 3,
        'class_name': 'thing',
        'class_source': 'vlm',
        'class_validated': False,
        **kw,
    }


def test_vlm_dismiss_stops_the_suggestion() -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _vlm_doc('c1')}})
    client = _client(fake)
    r = client.post('/curation/crops/c1/vlm_dismiss')
    assert r.status_code == 200, r.text
    item = r.json()
    assert item['vlm_proposed_class_id'] is None
    assert item['vlm_proposed_class_name'] is None
    assert item['proposed_class_id'] is None
    doc = fake.docs(ITEMS)['c1']
    assert doc['vlm_dismissed_class_id'] == 3
    assert doc['vlm_dismissed_at']


def test_vlm_dismiss_of_a_new_class_proposal() -> None:
    doc = {
        'crop_id': 'n1',
        'class_source': 'vlm_new_class_pending',
        'vlm_proposed_class': 'kayak',
    }
    fake = QueryFakeOpenSearch({ITEMS: {'n1': doc}})
    item = _client(fake).post('/curation/crops/n1/vlm_dismiss').json()
    assert item['vlm_proposed_class_name'] is None
    assert item['proposed_class_name'] == ''


def test_a_different_later_suggestion_is_shown_again() -> None:
    doc = _vlm_doc('c1', class_id=5, class_name='other', vlm_dismissed_class_id=3)
    item = serialize_item(doc, 'c1', api_prefix='')
    assert item['vlm_proposed_class_id'] == 5


def test_vlm_dismiss_without_a_suggestion_is_409() -> None:
    fake = QueryFakeOpenSearch(
        {ITEMS: {'h': {'crop_id': 'h', 'class_id': 1, 'class_source': 'human'}}}
    )
    client = _client(fake)
    assert client.post('/curation/crops/h/vlm_dismiss').status_code == 409
    assert client.post('/curation/crops/nope/vlm_dismiss').status_code == 404


def test_history_returns_sanitized_entries() -> None:
    history = [
        {
            'class_id': 1,
            'class_name': 'a',
            'class_source': 'vlm',
            'label_source': 'vlm',
            'class_validated': False,
            'cluster_id': 1,
            'writer': 'human:label_crop',
            'at': '2026-09-01T00:00:00+00:00',
            'restorable': True,
            'internal_blob': {'x': 1},
        }
    ]
    fake = QueryFakeOpenSearch({ITEMS: {'c1': {'crop_id': 'c1', 'class_id_history': history}}})
    client = _client(fake)
    r = client.get('/curation/crops/c1/history')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['crop_id'] == 'c1'
    (entry,) = body['entries']
    assert 'internal_blob' not in entry
    assert entry['writer'] == 'human:label_crop'
    assert entry['class_name'] == 'a'
    assert client.get('/curation/crops/nope/history').status_code == 404


def test_suspected_fp_threshold_defaults_server_side(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.routers.curation import regions_fp

    class _Store:
        metadata: dict[str, Any] = {}

        def load(self) -> bool:
            return False

    monkeypatch.setattr('src.services.detection.fp_store.FalsePositiveCentroidStore', _Store)
    r = _client(QueryFakeOpenSearch({ITEMS: {}})).get('/curation/regions/suspected_false_positives')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['threshold'] == regions_fp.SUSPECTED_FP_MAX_DISTANCE
    assert body['default_threshold'] == regions_fp.SUSPECTED_FP_MAX_DISTANCE
