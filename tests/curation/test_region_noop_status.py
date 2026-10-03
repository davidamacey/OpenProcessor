"""A status write that doesn't change the stored status is a no-op for the
region's derived state.

Live case (crop 938e2635...): a region already ``false_positive`` with
``region_verified=true`` (written before the status invariants existed)
was included in a bulk "false positive" write. Its status didn't change,
yet ``region_verified`` flipped true -> false and its region-cluster
placement was re-written. A re-assertion of the stored status must leave
``region_verified`` and the box's region-cluster fields alone; only a real
status change re-derives them.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_regions_router import _FakeRegionOS
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.curation.region_writes import human_status_box_write


# No-profile gating contract: this file exercises region routes, which
# require an active region profile (409 otherwise).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


F = get_region_fields()
BOX = (0.1, 0.2, 0.3, 0.4)


def _item(crop_id: str, status: RegionStatus, box: RegionBox, *, verified: bool) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'bbox_norm': [0.0, 0.0, 0.5, 0.5],
        F.status: status.value,
        F.verified: verified,
        **boxes_write_fields([box], current_src={}),
    }


def _fp_box(**over: Any) -> RegionBox:
    return RegionBox(
        box_id='b1',
        bbox_norm=BOX,
        state='false_positive',
        score=0.74,
        cluster_id=FALSE_POSITIVE_REGION_CLUSTER_ID,
        cluster_subid='-100b',
        cluster_distance=0.12,
        **over,
    )


def _accepted_box() -> RegionBox:
    return RegionBox(
        box_id='b1',
        bbox_norm=BOX,
        state='accepted',
        score=0.9,
        cluster_id=7,
        cluster_subid='7a',
        cluster_distance=0.2,
    )


@pytest.fixture
def fake_os() -> _FakeRegionOS:
    return _FakeRegionOS(
        {
            'fp-1': _item('fp-1', RegionStatus.FALSE_POSITIVE, _fp_box(), verified=True),
            'det-1': _item('det-1', RegionStatus.DETECTED, _accepted_box(), verified=True),
        }
    )


@pytest.fixture
def client(fake_os: _FakeRegionOS) -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


def test_bulk_false_positive_on_false_positive_keeps_verified(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = client.post(
        '/curation/projects/default/regions/batch_status',
        json={'crop_ids': ['fp-1'], 'region_status': 'false_positive'},
    )
    assert resp.status_code == 200, resp.text
    doc = fake_os._docs['fp-1']
    assert doc[F.verified] is True
    # The box is already a false positive: its FP sub-type placement (the
    # sub-id and the distance the matcher measured) is not re-written.
    (box,) = doc[F.boxes]
    assert box['cluster_id'] == FALSE_POSITIVE_REGION_CLUSTER_ID
    assert box['cluster_subid'] == '-100b'
    assert box['cluster_distance'] == 0.12
    assert resp.json()['items'][0]['region_verified'] is True


def test_patch_same_status_keeps_verified(client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = client.patch(
        '/curation/projects/default/crops/fp-1/region_meta',
        json={'region_status': 'false_positive'},
    )
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['fp-1'][F.verified] is True


def test_reconfirm_detected_keeps_region_cluster(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = client.post(
        '/curation/projects/default/regions/batch_status',
        json={'crop_ids': ['det-1'], 'region_status': 'detected'},
    )
    assert resp.status_code == 200, resp.text
    doc = fake_os._docs['det-1']
    (box,) = doc[F.boxes]
    assert box['cluster_id'] == 7
    assert box['cluster_subid'] == '7a'
    assert doc[F.verified] is True


def test_status_change_still_derives_verified_and_parks_the_box_in_the_fp_cluster() -> None:
    current = _item('x', RegionStatus.DETECTED, _accepted_box(), verified=True)
    doc = human_status_box_write('false_positive', current)
    assert doc[F.verified] is False
    (box,) = doc[F.boxes]
    assert box['state'] == 'false_positive'
    assert box['cluster_id'] == FALSE_POSITIVE_REGION_CLUSTER_ID
    assert box['cluster_subid'] is None
    assert box['cluster_distance'] == 0.0


def test_confirm_same_status_sets_verified_when_unverified() -> None:
    current = _item('x', RegionStatus.DETECTED, _accepted_box(), verified=False)
    doc = human_status_box_write('detected', current)
    assert doc[F.verified] is True
    (box,) = doc[F.boxes]
    assert box['cluster_id'] == 7


def test_a_whole_set_reject_by_an_automated_source_does_not_take_the_human_lock() -> None:
    from src.services.curation.region_boxes import is_human_owned, read_boxes

    current = _item('x', RegionStatus.DETECTED, _accepted_box(), verified=True)
    machine = human_status_box_write('verify_rejected', current, label_source='vlm_relabel')
    human = human_status_box_write('verify_rejected', current, label_source='human')
    assert not any(is_human_owned(b) for b in read_boxes(machine, F))
    assert all(is_human_owned(b) for b in read_boxes(human, F))
