"""A human "confirm" that doesn't move the box keeps detector provenance.

Live case (crop 2b8f1f7f...): the labeler confirmed a detector's region
with ``PUT /crops/{id}/region`` carrying the unchanged box; the write
re-stamped ``region_detector='human'`` / ``region_score=1.0`` and the
detector's name and 0.894 score were gone. A PUT whose box equals the
stored box (within float noise) records the human confirmation only —
status / verified / validated / verifier — and leaves the detector, its
version, score and detection time alone. A moved box is human geometry,
as before.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config, get_region_fields
from src.services.detection.profile_registry import region_profile_or_neutral


# No-profile gating contract: this file exercises region routes, which
# require an active region profile (409 otherwise).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


F = get_region_fields()
INDEX = get_curation_config().items_index
BOX = [0.6430511474609375, 0.5797716302772215, 0.6966705322265625, 0.6243825534416916]
PARENT = [0.2421875, 0.38985655737704916, 0.7890625, 0.8000585480093677]
HUMAN = region_profile_or_neutral().human_detector_name


def _detected(crop_id: str) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'bbox_norm': list(PARENT),
        F.bbox_norm: list(BOX),
        F.score: 0.89404296875,
        F.status: 'detected',
        F.verified: True,
        F.validated: False,
        F.detector: 'det_model',
        F.detector_version: '1',
        F.detected_at: '2026-09-24T03:08:19.221513+00:00',
        F.verifier: 'vlm_model',
        F.bbox_frame: 'source',
        F.cluster_id: 3,
    }


@pytest.fixture
def fake_os() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch({INDEX: {'c1': _detected('c1'), 'c2': _detected('c2')}})


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


def _assert_provenance_kept(doc: dict[str, Any]) -> None:
    assert doc[F.detector] == 'det_model'
    assert doc[F.detector_version] == '1'
    assert doc[F.score] == 0.89404296875
    assert doc[F.detected_at] == '2026-09-24T03:08:19.221513+00:00'
    assert doc[F.bbox_norm] == BOX


def test_same_box_put_keeps_detector_and_score(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    resp = client.put('/curation/crops/c1/region', json={'region_bbox_norm': BOX})
    assert resp.status_code == 200, resp.text
    doc = _doc(fake_os, 'c1')
    _assert_provenance_kept(doc)
    assert doc[F.status] == 'detected'
    assert doc[F.verified] is True
    assert doc[F.validated] is True
    assert doc[F.verifier] == HUMAN
    assert doc[F.label_source] == 'human'
    item = resp.json()['item']
    assert item['region_detector'] == 'det_model'
    assert item['region_score'] == 0.89404296875


def test_same_box_within_float_noise(client: TestClient, fake_os: QueryFakeOpenSearch) -> None:
    noisy = [v + 3e-7 for v in BOX]
    resp = client.put('/curation/crops/c1/region', json={'region_bbox_norm': noisy})
    assert resp.status_code == 200, resp.text
    _assert_provenance_kept(_doc(fake_os, 'c1'))


def test_same_box_in_parent_frame(client: TestClient, fake_os: QueryFakeOpenSearch) -> None:
    px1, py1, px2, py2 = PARENT
    w, h = px2 - px1, py2 - py1
    in_parent = [
        (BOX[0] - px1) / w,
        (BOX[1] - py1) / h,
        (BOX[2] - px1) / w,
        (BOX[3] - py1) / h,
    ]
    resp = client.put(
        '/curation/crops/c1/region', json={'region_bbox_norm': in_parent, 'frame': 'parent'}
    )
    assert resp.status_code == 200, resp.text
    _assert_provenance_kept(_doc(fake_os, 'c1'))


def test_same_box_batch_put_keeps_provenance(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    resp = client.put(
        '/curation/crops/batch_region', json={'crop_ids': ['c1', 'c2'], 'region_bbox_norm': BOX}
    )
    assert resp.status_code == 200, resp.text
    for cid in ('c1', 'c2'):
        _assert_provenance_kept(_doc(fake_os, cid))


def test_same_box_confirm_of_false_positive_confirms(
    client: TestClient, fake_os: QueryFakeOpenSearch
) -> None:
    _doc(fake_os, 'c1').update({F.status: 'false_positive', F.verified: False, F.cluster_id: -100})
    resp = client.put('/curation/crops/c1/region', json={'region_bbox_norm': BOX})
    assert resp.status_code == 200, resp.text
    doc = _doc(fake_os, 'c1')
    _assert_provenance_kept(doc)
    assert doc[F.status] == 'detected'
    assert doc[F.verified] is True
    assert doc[F.cluster_id] is None


def test_moved_box_is_human_geometry(client: TestClient, fake_os: QueryFakeOpenSearch) -> None:
    moved = [BOX[0] + 0.01, BOX[1], BOX[2] + 0.01, BOX[3]]
    resp = client.put('/curation/crops/c1/region', json={'region_bbox_norm': moved})
    assert resp.status_code == 200, resp.text
    doc = _doc(fake_os, 'c1')
    assert doc[F.detector] == HUMAN
    assert doc[F.score] == 1.0
    assert doc[F.bbox_norm] == moved
