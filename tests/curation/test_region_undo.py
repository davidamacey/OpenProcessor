"""Region writes and VLM dismissals are undoable.

Every human region writer snapshots the pre-write region state into the
item's edit history; ``POST /crops/{id}/region/undo`` (and the batch form)
put it back exactly — box, score, status, flags, detector provenance and
region-cluster placement — and step back through successive writes.
``POST /crops/{id}/vlm_dismiss/undo`` brings a dismissed VLM suggestion
back.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config, get_region_fields


F = get_region_fields()
INDEX = get_curation_config().items_index
BOX = [0.2, 0.2, 0.4, 0.4]


def _detected(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'bbox_norm': [0.0, 0.0, 0.5, 0.5],
        F.bbox_norm: list(BOX),
        F.score: 0.894,
        F.status: 'detected',
        F.verified: True,
        F.validated: False,
        F.detector: 'det_model',
        F.detector_version: '3',
        F.verifier: 'vlm_model',
        F.detected_at: '2026-09-24T03:08:19+00:00',
        F.cluster_id: 5,
        F.cluster_subid: '5a',
        **extra,
    }


def _state(doc: dict[str, Any]) -> dict[str, Any]:
    keys = (
        F.bbox_norm,
        F.score,
        F.status,
        F.verified,
        F.validated,
        F.detector,
        F.detector_version,
        F.verifier,
        F.detected_at,
        F.cluster_id,
        F.cluster_subid,
        F.label_source,
    )
    return {k: doc.get(k) for k in keys}


@pytest.fixture
def fake_os() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            INDEX: {
                'c1': _detected('c1'),
                'c2': _detected('c2'),
                'fresh': _detected('fresh'),
                'vlm-1': {
                    'crop_id': 'vlm-1',
                    'class_id': 3,
                    'class_name': 'alpha',
                    'class_source': 'vlm',
                    'label_source': 'vlm',
                },
            }
        }
    )


@pytest.fixture
def client(fake_os: QueryFakeOpenSearch) -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


def _doc(fake_os: QueryFakeOpenSearch, crop_id: str) -> dict[str, Any]:
    return fake_os.docs(INDEX)[crop_id]


def test_undo_box_edit_restores_detector_provenance(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    before = _state(_doc(fake_os, 'c1'))
    resp = client.put('/curation/crops/c1/region', json={'region_bbox_norm': [0.1, 0.1, 0.3, 0.3]})
    assert resp.status_code == 200, resp.text
    assert _doc(fake_os, 'c1')[F.detector] != 'det_model'

    resp = client.post('/curation/crops/c1/region/undo')
    assert resp.status_code == 200, resp.text
    assert _state(_doc(fake_os, 'c1')) == before
    assert resp.json()['region_score'] == 0.894
    assert resp.json()['region_bbox_norm'] == BOX


@pytest.mark.parametrize(
    'body',
    [
        {'region_status': 'false_positive'},
        {'region_status': 'no_region_visible'},
        {'region_status': 'verify_rejected', 'region_rejection_reason': 'blurry'},
    ],
)
def test_undo_status_write(
    client: TestClient, fake_os: QueryFakeOpenSearch, body: dict[str, Any]
) -> None:
    before = _state(_doc(fake_os, 'c1'))
    assert client.patch('/curation/crops/c1/region_meta', json=body).status_code == 200
    assert _state(_doc(fake_os, 'c1')) != before
    resp = client.post('/curation/crops/c1/region/undo')
    assert resp.status_code == 200, resp.text
    assert _state(_doc(fake_os, 'c1')) == before
    assert _doc(fake_os, 'c1').get(F.rejection_reason) is None


def test_repeated_undo_steps_back(client: TestClient, fake_os: QueryFakeOpenSearch) -> None:
    original = _state(_doc(fake_os, 'c1'))
    client.patch('/curation/crops/c1/region_meta', json={'region_status': 'false_positive'})
    after_fp = _state(_doc(fake_os, 'c1'))
    client.put('/curation/crops/c1/region', json={'region_bbox_norm': None})

    assert client.post('/curation/crops/c1/region/undo').status_code == 200
    assert _state(_doc(fake_os, 'c1')) == after_fp
    assert client.post('/curation/crops/c1/region/undo').status_code == 200
    assert _state(_doc(fake_os, 'c1')) == original
    assert client.post('/curation/crops/c1/region/undo').status_code == 409


def test_undo_with_no_region_write_is_409(client: TestClient) -> None:
    assert client.post('/curation/crops/fresh/region/undo').status_code == 409


def test_undo_unknown_crop_is_404(client: TestClient) -> None:
    assert client.post('/curation/crops/nope/region/undo').status_code == 404


def test_undo_batch_reports_per_crop_outcomes(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    before = {cid: _state(_doc(fake_os, cid)) for cid in ('c1', 'c2')}
    resp = client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['c1', 'c2'], 'region_status': 'false_positive'},
    )
    assert resp.status_code == 200, resp.text

    resp = client.post(
        '/curation/crops/region/undo_batch',
        json={'crop_ids': ['c1', 'c2', 'fresh', 'nope']},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['undone'] == 2
    assert sorted(i['crop_id'] for i in body['items']) == ['c1', 'c2']
    assert body['nothing_to_undo'] == ['fresh']
    assert body['not_found'] == ['nope']
    assert body['conflicts'] == []
    for cid in ('c1', 'c2'):
        assert _state(_doc(fake_os, cid)) == before[cid]


def test_undo_batch_of_bulk_box_clear(client: TestClient, fake_os: QueryFakeOpenSearch) -> None:
    before = _state(_doc(fake_os, 'c2'))
    client.put('/curation/crops/batch_region', json={'crop_ids': ['c2'], 'region_bbox_norm': None})
    resp = client.post('/curation/crops/region/undo_batch', json={'crop_ids': ['c2']})
    assert resp.status_code == 200, resp.text
    assert _state(_doc(fake_os, 'c2')) == before


def test_undo_batch_nothing_anywhere_is_409(client: TestClient) -> None:
    resp = client.post('/curation/crops/region/undo_batch', json={'crop_ids': ['fresh']})
    assert resp.status_code == 409


def test_region_undo_leaves_class_untouched(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    _doc(fake_os, 'c1').update({'class_id': 4, 'class_source': 'human', 'class_validated': True})
    client.patch('/curation/crops/c1/region_meta', json={'region_status': 'false_positive'})
    client.post('/curation/crops/c1/region/undo')
    doc = _doc(fake_os, 'c1')
    assert (doc['class_id'], doc['class_source'], doc['class_validated']) == (4, 'human', True)


# ---------------------------------------------------------------- vlm dismiss


def test_vlm_dismiss_undo_restores_suggestion(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    resp = client.post('/curation/crops/vlm-1/vlm_dismiss')
    assert resp.status_code == 200, resp.text
    assert _doc(fake_os, 'vlm-1')['vlm_dismissed_class_name'] == 'alpha'

    resp = client.post('/curation/crops/vlm-1/vlm_dismiss/undo')
    assert resp.status_code == 200, resp.text
    doc = _doc(fake_os, 'vlm-1')
    assert doc.get('vlm_dismissed_class_name') is None
    assert doc.get('vlm_dismissed_class_id') is None
    assert resp.json()['vlm_proposed_class_name'] == 'alpha'
    assert client.post('/curation/crops/vlm-1/vlm_dismiss/undo').status_code == 409


def test_vlm_dismiss_undo_without_dismissal_is_409(client: TestClient) -> None:
    assert client.post('/curation/crops/vlm-1/vlm_dismiss/undo').status_code == 409
