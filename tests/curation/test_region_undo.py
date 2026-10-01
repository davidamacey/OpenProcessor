"""Region writes and VLM dismissals are undoable.

Every human region writer snapshots the pre-write region state into the
item's edit history; ``POST /crops/{id}/region/undo`` (and the batch form)
put it back exactly — the whole box list, status, flags and detector
provenance — and step back through successive writes. An undo is itself a
write: the revision moves forward and the box-id high-water mark never
decreases.
``POST /crops/{id}/vlm_dismiss/undo`` brings a dismissed VLM suggestion
back.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.services.curation.region_boxes import RegionBox, boxes_write_fields


# No-profile gating contract: this file exercises region routes, which
# require an active region profile (409 otherwise).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


F = get_region_fields()
INDEX = base_curation_config().items_index
BOX = [0.2, 0.2, 0.4, 0.4]
BOX_TUPLE = (0.2, 0.2, 0.4, 0.4)


def _detected(crop_id: str, **extra: Any) -> dict[str, Any]:
    box = RegionBox(
        box_id='b1',
        bbox_norm=BOX_TUPLE,
        state='accepted',
        score=0.894,
        detector='det_model',
        detector_version='3',
        source='detector',
        cluster_id=5,
        cluster_subid='5a',
        detected_at='2026-09-24T03:08:19+00:00',
    )
    return {
        'crop_id': crop_id,
        'bbox_norm': [0.0, 0.0, 0.5, 0.5],
        F.status: 'detected',
        F.verified: True,
        F.validated: False,
        F.verifier: 'vlm_model',
        F.detected_at: '2026-09-24T03:08:19+00:00',
        **boxes_write_fields([box], current_src={}),
        **extra,
    }


def _state(doc: dict[str, Any]) -> dict[str, Any]:
    keys = (
        F.boxes,
        F.count,
        F.rejected_count,
        F.max_score,
        F.status,
        F.verified,
        F.validated,
        F.verifier,
        F.detected_at,
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
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


def _doc(fake_os: QueryFakeOpenSearch, crop_id: str) -> dict[str, Any]:
    return fake_os.docs(INDEX)[crop_id]


def test_undo_box_edit_restores_detector_provenance(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    before = _state(_doc(fake_os, 'c1'))
    resp = client.put(
        '/curation/projects/default/crops/c1/regions',
        json={'boxes': [{'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.3, 0.3]}]},
    )
    assert resp.status_code == 200, resp.text
    assert _doc(fake_os, 'c1')[F.boxes][0]['detector'] != 'det_model'

    resp = client.post('/curation/projects/default/crops/c1/region/undo')
    assert resp.status_code == 200, resp.text
    assert _state(_doc(fake_os, 'c1')) == before
    (box,) = resp.json()['region_boxes']
    assert box['score'] == 0.894
    assert box['bbox_norm'] == BOX
    assert box['detector'] == 'det_model'


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
    assert (
        client.patch('/curation/projects/default/crops/c1/region_meta', json=body).status_code
        == 200
    )
    assert _state(_doc(fake_os, 'c1')) != before
    resp = client.post('/curation/projects/default/crops/c1/region/undo')
    assert resp.status_code == 200, resp.text
    assert _state(_doc(fake_os, 'c1')) == before
    assert _doc(fake_os, 'c1').get(F.rejection_reason) is None


def test_undo_restores_the_whole_box_list_after_a_multi_box_edit(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    """W8 pin 4: one PUT that moves a box, adds one and deletes another is
    one undo step that restores the whole prior list."""
    client.put(
        '/curation/projects/default/crops/c1/regions',
        json={
            'boxes': [
                {'box_id': 'b1'},
                {'box_id': None, 'bbox_norm': [0.6, 0.6, 0.7, 0.7]},
                {'box_id': None, 'bbox_norm': [0.7, 0.1, 0.8, 0.2]},
            ]
        },
    )
    before_boxes = _doc(fake_os, 'c1')[F.boxes]
    assert [b['box_id'] for b in before_boxes] == ['b1', 'b2', 'b3']
    resp = client.put(
        '/curation/projects/default/crops/c1/regions',
        json={
            'boxes': [
                {'box_id': 'b1', 'bbox_norm': [0.15, 0.15, 0.25, 0.25]},
                {'box_id': 'b3'},
                {'box_id': None, 'bbox_norm': [0.05, 0.05, 0.15, 0.15]},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    assert [b['box_id'] for b in _doc(fake_os, 'c1')[F.boxes]] == ['b1', 'b3', 'b4']

    resp = client.post('/curation/projects/default/crops/c1/region/undo')
    assert resp.status_code == 200, resp.text
    restored = _doc(fake_os, 'c1')
    assert restored[F.boxes] == before_boxes
    assert restored[F.count] == 3


def test_undo_moves_the_revision_forward_and_never_reuses_a_box_id(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    rev0 = _doc(fake_os, 'c1')[F.revision]
    client.put(
        '/curation/projects/default/crops/c1/regions',
        json={'boxes': [{'box_id': 'b1'}, {'box_id': None, 'bbox_norm': [0.6, 0.6, 0.7, 0.7]}]},
    )
    assert _doc(fake_os, 'c1')[F.revision] == rev0 + 1
    assert client.post('/curation/projects/default/crops/c1/region/undo').status_code == 200
    doc = _doc(fake_os, 'c1')
    assert [b['box_id'] for b in doc[F.boxes]] == ['b1']
    # The undo is itself a write: a client holding the pre-undo revision
    # is stale, and the restored snapshot's own (older) revision is not
    # what the item reports.
    assert doc[F.revision] == rev0 + 2
    stale = client.put(
        '/curation/projects/default/crops/c1/regions',
        json={'boxes': [{'box_id': 'b1'}], 'expected_region_revision': rev0 + 1},
    )
    assert stale.status_code == 409, stale.text
    # b2 was undone away; the next new box must not reuse its id.
    resp = client.put(
        '/curation/projects/default/crops/c1/regions',
        json={'boxes': [{'box_id': 'b1'}, {'box_id': None, 'bbox_norm': [0.6, 0.6, 0.7, 0.7]}]},
    )
    assert resp.status_code == 200, resp.text
    assert [b['box_id'] for b in _doc(fake_os, 'c1')[F.boxes]] == ['b1', 'b3']


def test_repeated_undo_steps_back(client: TestClient, fake_os: QueryFakeOpenSearch) -> None:
    original = _state(_doc(fake_os, 'c1'))
    client.patch(
        '/curation/projects/default/crops/c1/region_meta', json={'region_status': 'false_positive'}
    )
    after_fp = _state(_doc(fake_os, 'c1'))
    client.put('/curation/projects/default/crops/c1/regions', json={'boxes': []})

    assert client.post('/curation/projects/default/crops/c1/region/undo').status_code == 200
    assert _state(_doc(fake_os, 'c1')) == after_fp
    assert client.post('/curation/projects/default/crops/c1/region/undo').status_code == 200
    assert _state(_doc(fake_os, 'c1')) == original
    assert client.post('/curation/projects/default/crops/c1/region/undo').status_code == 409


def test_undo_with_no_region_write_is_409(client: TestClient) -> None:
    assert client.post('/curation/projects/default/crops/fresh/region/undo').status_code == 409


def test_undo_unknown_crop_is_404(client: TestClient) -> None:
    assert client.post('/curation/projects/default/crops/nope/region/undo').status_code == 404


def test_undo_batch_reports_per_crop_outcomes(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    before = {cid: _state(_doc(fake_os, cid)) for cid in ('c1', 'c2')}
    resp = client.post(
        '/curation/projects/default/regions/batch_status',
        json={'crop_ids': ['c1', 'c2'], 'region_status': 'false_positive'},
    )
    assert resp.status_code == 200, resp.text

    resp = client.post(
        '/curation/projects/default/crops/region/undo_batch',
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
    client.put(
        '/curation/projects/default/crops/batch_regions',
        json={'crop_ids': ['c2'], 'boxes': []},
    )
    resp = client.post(
        '/curation/projects/default/crops/region/undo_batch', json={'crop_ids': ['c2']}
    )
    assert resp.status_code == 200, resp.text
    assert _state(_doc(fake_os, 'c2')) == before


def test_undo_batch_nothing_anywhere_is_409(client: TestClient) -> None:
    resp = client.post(
        '/curation/projects/default/crops/region/undo_batch', json={'crop_ids': ['fresh']}
    )
    assert resp.status_code == 409


def test_region_undo_leaves_class_untouched(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    _doc(fake_os, 'c1').update({'class_id': 4, 'class_source': 'human', 'class_validated': True})
    client.patch(
        '/curation/projects/default/crops/c1/region_meta', json={'region_status': 'false_positive'}
    )
    client.post('/curation/projects/default/crops/c1/region/undo')
    doc = _doc(fake_os, 'c1')
    assert (doc['class_id'], doc['class_source'], doc['class_validated']) == (4, 'human', True)


# ---------------------------------------------------------------- vlm dismiss


def test_vlm_dismiss_undo_restores_suggestion(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    resp = client.post('/curation/projects/default/crops/vlm-1/vlm_dismiss')
    assert resp.status_code == 200, resp.text
    assert _doc(fake_os, 'vlm-1')['vlm_dismissed_class_name'] == 'alpha'

    resp = client.post('/curation/projects/default/crops/vlm-1/vlm_dismiss/undo')
    assert resp.status_code == 200, resp.text
    doc = _doc(fake_os, 'vlm-1')
    assert doc.get('vlm_dismissed_class_name') is None
    assert doc.get('vlm_dismissed_class_id') is None
    assert resp.json()['vlm_proposed_class_name'] == 'alpha'
    assert client.post('/curation/projects/default/crops/vlm-1/vlm_dismiss/undo').status_code == 409


def test_vlm_dismiss_undo_without_dismissal_is_409(client: TestClient) -> None:
    assert client.post('/curation/projects/default/crops/vlm-1/vlm_dismiss/undo').status_code == 409
