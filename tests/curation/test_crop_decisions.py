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

from typing import Any

import pytest
from _embedding_registry_fake import DEFAULT_STATE_FIELDS
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation.wire import serialize_item


ITEMS = base_curation_config().items_index


def _client(fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
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
    r = client.post('/curation/projects/default/crops/c1/vlm_dismiss')
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
    item = _client(fake).post('/curation/projects/default/crops/n1/vlm_dismiss').json()
    assert item['vlm_proposed_class_name'] is None
    assert item['proposed_class_name'] == ''


def test_a_different_later_suggestion_is_shown_again() -> None:
    doc = _vlm_doc('c1', class_id=5, class_name='other', vlm_dismissed_class_id=3)
    item = serialize_item(doc, 'c1', state_fields=DEFAULT_STATE_FIELDS, api_prefix='')
    assert item['vlm_proposed_class_id'] == 5


def test_vlm_dismiss_without_a_suggestion_is_409() -> None:
    fake = QueryFakeOpenSearch(
        {ITEMS: {'h': {'crop_id': 'h', 'class_id': 1, 'class_source': 'human'}}}
    )
    client = _client(fake)
    assert client.post('/curation/projects/default/crops/h/vlm_dismiss').status_code == 409
    assert client.post('/curation/projects/default/crops/nope/vlm_dismiss').status_code == 404


def _probe_doc(crop_id: str, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_id': 1,
        'class_name': 'bus',
        'class_source': 'human',
        'class_validated': True,
        'probe_pred_class': 'car',
        'probe_disagreement': True,
        'probe_model_version': 'run-1',
        **kw,
    }


def test_probe_dismiss_keeps_the_label_and_records_history() -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'p1': _probe_doc('p1')}})
    client = _client(fake)
    r = client.post('/curation/projects/default/crops/p1/probe_dismiss')
    assert r.status_code == 200, r.text
    doc = fake.docs(ITEMS)['p1']
    assert doc['class_name'] == 'bus'
    assert doc['probe_dismissed_class'] == 'car'
    assert doc['probe_dismissed_model_version'] == 'run-1'
    assert doc['probe_dismissed_at']
    assert doc['edit_history'][-1]['kind'] == 'probe_dismiss'
    assert doc['edit_history'][-1]['writer'] == 'human:probe_dismiss'


def test_probe_dismiss_undo_restores_the_disagreement() -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'p1': _probe_doc('p1')}})
    client = _client(fake)
    client.post('/curation/projects/default/crops/p1/probe_dismiss')
    r = client.post('/curation/projects/default/crops/p1/probe_dismiss/undo')
    assert r.status_code == 200, r.text
    assert fake.docs(ITEMS)['p1'].get('probe_dismissed_at') is None
    again = client.post('/curation/projects/default/crops/p1/probe_dismiss/undo')
    assert again.status_code == 409


def test_probe_dismiss_without_a_disagreement_is_409() -> None:
    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'agrees': _probe_doc('agrees', probe_disagreement=False),
                'blind': _probe_doc('blind', probe_disagreement=None),
            }
        }
    )
    client = _client(fake)
    assert client.post('/curation/projects/default/crops/agrees/probe_dismiss').status_code == 409
    assert client.post('/curation/projects/default/crops/blind/probe_dismiss').status_code == 409
    assert client.post('/curation/projects/default/crops/nope/probe_dismiss').status_code == 404


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
    r = client.get('/curation/projects/default/crops/c1/history')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['crop_id'] == 'c1'
    (entry,) = body['entries']
    assert 'internal_blob' not in entry
    assert entry['writer'] == 'human:label_crop'
    assert entry['class_name'] == 'a'
    assert client.get('/curation/projects/default/crops/nope/history').status_code == 404


@pytest.mark.usefixtures('reference_region_profile')
def test_suspected_fp_threshold_defaults_server_side(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.routers.curation import regions_fp

    class _Store:
        metadata: dict[str, Any] = {}

        def load(self) -> bool:
            return False

    monkeypatch.setattr('src.services.detection.fp_store.FalsePositiveCentroidStore', _Store)
    r = _client(QueryFakeOpenSearch({ITEMS: {}})).get(
        '/curation/projects/default/regions/suspected_false_positives'
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['threshold'] == regions_fp.SUSPECTED_FP_MAX_DISTANCE
    assert body['default_threshold'] == regions_fp.SUSPECTED_FP_MAX_DISTANCE
