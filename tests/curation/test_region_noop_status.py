"""A status write that doesn't change the stored status is a no-op for the
region's derived state.

Live case (crop 938e2635...): a region already ``false_positive`` with
``region_verified=true`` (written before the status invariants existed)
was included in a bulk "false positive" write. Its status didn't change,
yet ``region_verified`` flipped true -> false and its region-cluster
placement was re-written. A re-assertion of the stored status must leave
``region_verified`` and the region-cluster fields alone; only a real status
change re-derives them.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_regions_router import _FakeRegionOS
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.services.curation.region_writes import human_status_fields


# No-profile gating contract: this file exercises region routes, which
# require an active region profile (409 otherwise).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


F = get_region_fields()
BOX = [0.1, 0.2, 0.3, 0.4]


def _fp_verified(crop_id: str) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'bbox_norm': [0.0, 0.0, 0.5, 0.5],
        F.bbox_norm: list(BOX),
        F.score: 0.74,
        F.status: RegionStatus.FALSE_POSITIVE.value,
        F.verified: True,
        F.cluster_id: -100,
        F.cluster_distance: 0.0,
    }


@pytest.fixture
def fake_os() -> _FakeRegionOS:
    return _FakeRegionOS(
        {
            'fp-1': _fp_verified('fp-1'),
            'det-1': {
                'crop_id': 'det-1',
                'bbox_norm': [0.0, 0.0, 0.5, 0.5],
                F.bbox_norm: list(BOX),
                F.score: 0.9,
                F.status: RegionStatus.DETECTED.value,
                F.verified: True,
                F.cluster_id: 7,
                F.cluster_subid: '7a',
            },
        }
    )


@pytest.fixture
def client(fake_os: _FakeRegionOS) -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


def test_bulk_false_positive_on_false_positive_keeps_verified(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['fp-1'], 'region_status': 'false_positive'},
    )
    assert resp.status_code == 200, resp.text
    doc = fake_os._docs['fp-1']
    assert doc[F.verified] is True
    assert doc[F.cluster_id] == -100
    assert resp.json()['items'][0]['region_verified'] is True


def test_patch_same_status_keeps_verified(client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = client.patch(
        '/curation/crops/fp-1/region_meta', json={'region_status': 'false_positive'}
    )
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['fp-1'][F.verified] is True


def test_reconfirm_detected_keeps_region_cluster(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['det-1'], 'region_status': 'detected'},
    )
    assert resp.status_code == 200, resp.text
    doc = fake_os._docs['det-1']
    assert doc[F.cluster_id] == 7
    assert doc[F.cluster_subid] == '7a'
    assert doc[F.verified] is True


def test_status_change_still_derives_verified() -> None:
    doc = human_status_fields('false_positive', {F.status: 'detected', F.verified: True})
    assert doc[F.verified] is False
    assert doc[F.cluster_id] == -100


def test_confirm_same_status_sets_verified_when_unverified() -> None:
    doc = human_status_fields(
        'detected', {F.status: 'detected', F.verified: False, F.bbox_norm: BOX}
    )
    assert doc[F.verified] is True
    assert F.cluster_id not in doc
