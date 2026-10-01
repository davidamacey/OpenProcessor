"""A human "confirm" that doesn't move the box keeps detector provenance.

Live case (crop 2b8f1f7f...): the labeler confirmed a detector's region
with a write carrying the unchanged box; the write re-stamped the box as
``human`` / score ``1.0`` and the detector's name and 0.894 score were
gone. A ``PUT /crops/{id}/regions`` element whose box equals the stored
one (within float noise) records the human confirmation only -- status /
verified / validated / verifier -- and leaves the box's detector, version,
score, source and detection time alone, and keeps the *stored* coordinates
rather than the request's noisy copy. A moved box is human geometry.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.detection.profile_registry import region_profile_or_neutral


# No-profile gating contract: this file exercises region routes, which
# require an active region profile (409 otherwise).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


F = get_region_fields()
INDEX = base_curation_config().items_index
BOX = [0.6430511474609375, 0.5797716302772215, 0.6966705322265625, 0.6243825534416916]
PARENT = [0.2421875, 0.38985655737704916, 0.7890625, 0.8000585480093677]
HUMAN = region_profile_or_neutral().human_detector_name
DETECTED_AT = '2026-09-24T03:08:19.221513+00:00'


def _detector_box(**over: Any) -> RegionBox:
    kwargs: dict[str, Any] = {
        'box_id': 'b1',
        'bbox_norm': tuple(BOX),
        'state': 'accepted',
        'score': 0.89404296875,
        'detector': 'det_model',
        'detector_version': '1',
        'source': 'detector',
        'bbox_correct': True,
        'confidence': 'high',
        'detected_at': DETECTED_AT,
    }
    kwargs.update(over)
    return RegionBox(**kwargs)


def _detected(crop_id: str, box: RegionBox | None = None) -> dict[str, Any]:
    box = box or _detector_box()
    return {
        'crop_id': crop_id,
        'bbox_norm': list(PARENT),
        F.status: 'detected' if box.state == 'accepted' else 'false_positive',
        F.verified: box.state == 'accepted',
        F.validated: False,
        F.verifier: 'vlm_model',
        **boxes_write_fields([box], current_src={}),
    }


@pytest.fixture
def fake_os() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch({INDEX: {'c1': _detected('c1'), 'c2': _detected('c2')}})


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


def _put(client: TestClient, boxes: list[dict[str, Any]], **body: Any) -> Any:
    return client.put(
        '/curation/projects/default/crops/c1/regions',
        json={'boxes': boxes, 'region_status': 'detected', **body},
    )


def _assert_box_provenance_kept(doc: dict[str, Any]) -> None:
    (box,) = doc[F.boxes]
    assert box['detector'] == 'det_model'
    assert box['detector_version'] == '1'
    assert box['score'] == 0.89404296875
    assert box['source'] == 'detector'
    assert box['detected_at'] == DETECTED_AT
    assert box['bbox_correct'] is True
    assert box['confidence'] == 'high'
    assert box['bbox_norm'] == BOX


def test_same_box_put_keeps_detector_and_score(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    resp = _put(client, [{'box_id': 'b1', 'bbox_norm': BOX}])
    assert resp.status_code == 200, resp.text
    doc = _doc(fake_os, 'c1')
    _assert_box_provenance_kept(doc)
    assert doc[F.status] == 'detected'
    assert doc[F.verified] is True
    assert doc[F.validated] is True
    assert doc[F.verifier] == HUMAN
    assert doc[F.label_source] == 'human'
    (wire_box,) = resp.json()['item']['region_boxes']
    assert wire_box['detector'] == 'det_model'
    assert wire_box['score'] == 0.89404296875


def test_same_box_within_float_noise_keeps_the_stored_coordinates(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    noisy = [v + 3e-7 for v in BOX]
    resp = _put(client, [{'box_id': 'b1', 'bbox_norm': noisy}])
    assert resp.status_code == 200, resp.text
    _assert_box_provenance_kept(_doc(fake_os, 'c1'))


def test_same_box_in_parent_frame(client: TestClient, fake_os: QueryFakeOpenSearch) -> None:
    px1, py1, px2, py2 = PARENT
    w, h = px2 - px1, py2 - py1
    in_parent = [
        (BOX[0] - px1) / w,
        (BOX[1] - py1) / h,
        (BOX[2] - px1) / w,
        (BOX[3] - py1) / h,
    ]
    resp = _put(client, [{'box_id': 'b1', 'bbox_norm': in_parent}], frame='parent')
    assert resp.status_code == 200, resp.text
    _assert_box_provenance_kept(_doc(fake_os, 'c1'))


def test_accepting_a_false_positive_box_confirms_it(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    fake_os.docs(INDEX)['c1'] = _detected(
        'c1',
        _detector_box(
            state='false_positive',
            cluster_id=FALSE_POSITIVE_REGION_CLUSTER_ID,
            cluster_distance=0.0,
        ),
    )
    resp = _put(client, [{'box_id': 'b1', 'bbox_norm': BOX, 'state': 'accepted'}])
    assert resp.status_code == 200, resp.text
    doc = _doc(fake_os, 'c1')
    _assert_box_provenance_kept(doc)
    (box,) = doc[F.boxes]
    assert box['state'] == 'accepted'
    # Leaving the false-positive bucket releases the box for re-clustering.
    assert box['cluster_id'] is None
    assert box['cluster_distance'] is None
    assert doc[F.status] == 'detected'
    assert doc[F.verified] is True


def test_whole_set_confirm_never_overrides_a_false_positive_decision(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    fake_os.docs(INDEX)['c1'] = _detected('c1', _detector_box(state='false_positive'))
    resp = _put(client, [{'box_id': 'b1'}])
    assert resp.status_code == 422, resp.text
    assert _doc(fake_os, 'c1')[F.boxes][0]['state'] == 'false_positive'


def test_moved_box_is_human_geometry(client: TestClient, fake_os: QueryFakeOpenSearch) -> None:
    moved = [BOX[0] + 0.01, BOX[1], BOX[2] + 0.01, BOX[3]]
    resp = _put(client, [{'box_id': 'b1', 'bbox_norm': moved}])
    assert resp.status_code == 200, resp.text
    doc = _doc(fake_os, 'c1')
    (box,) = doc[F.boxes]
    assert box['detector'] == HUMAN
    assert box['source'] == 'human'
    assert box['score'] == 1.0
    assert box['bbox_norm'] == moved
    # The verdict that judged the old geometry doesn't carry over.
    assert box['bbox_correct'] is None
    assert box['confidence'] is None
    assert box['state'] == 'accepted'
    assert doc[F.max_score] == 1.0
