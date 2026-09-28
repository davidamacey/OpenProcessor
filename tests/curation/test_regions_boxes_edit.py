"""Write-path tests for the W8a multi-box human edit routes:
``PUT /crops/{id}/regions``, ``PUT /crops/batch_regions``,
``PATCH /crops/{id}/regions/{box_id}``, ``POST /regions/batch_box_state``.

Drives the real FastAPI routes against a fake OpenSearch, reusing the
``_FakeRegionOS`` double from ``test_regions_router.py`` (same shape,
same call surface).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_regions_router import _FakeRegionOS
from src.config import get_region_fields


F = get_region_fields()

pytestmark = pytest.mark.usefixtures('reference_region_profile')


@pytest.fixture
def fake_os() -> _FakeRegionOS:
    return _FakeRegionOS(
        {
            'crop-1': {
                'crop_id': 'crop-1',
                F.status: 'pending_detection',
                F.bbox_norm: [0.1, 0.1, 0.2, 0.2],
                F.boxes: [
                    {
                        'box_id': 'b1',
                        'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                        'state': 'accepted',
                        'score': 0.9,
                    },
                    {
                        'box_id': 'b2',
                        'bbox_norm': [0.3, 0.3, 0.4, 0.4],
                        'state': 'proposed',
                        'score': 0.5,
                    },
                ],
                F.box_seq: 2,
                F.revision: 3,
                F.count: 1,
                F.rejected_count: 0,
            },
            'crop-2': {
                'crop_id': 'crop-2',
                F.status: 'pending_detection',
                F.boxes: [],
                F.box_seq: 0,
                F.revision: 0,
            },
        }
    )


@pytest.fixture
def app_client(fake_os: _FakeRegionOS) -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# PUT crops-id-regions
# ---------------------------------------------------------------------------


def test_put_regions_untouched_sibling_survives(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': 'b1'}, {'box_id': 'b2'}]},
    )
    assert resp.status_code == 200, resp.text
    stored = fake_os._docs['crop-1'][F.boxes]
    assert [b['box_id'] for b in stored] == ['b1', 'b2']
    assert stored[1]['state'] == 'proposed'


def test_put_regions_omitting_a_box_deletes_it(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': 'b1'}]},
    )
    assert resp.status_code == 200, resp.text
    stored = fake_os._docs['crop-1'][F.boxes]
    assert [b['box_id'] for b in stored] == ['b1']
    assert fake_os._docs['crop-1'][F.count] == 1


def test_put_regions_new_box_defaults_to_accepted(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={
            'boxes': [
                {'box_id': 'b1'},
                {'box_id': 'b2'},
                {'box_id': None, 'bbox_norm': [0.5, 0.5, 0.6, 0.6]},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    stored = fake_os._docs['crop-1'][F.boxes]
    assert stored[2]['box_id'] == 'b3'
    assert stored[2]['state'] == 'accepted'


def test_put_regions_with_region_status_confirms_only_proposed(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    """Enter confirms only proposed boxes -- b1 is already accepted and
    must stay accepted, not be re-derived from scratch."""
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': 'b1'}, {'box_id': 'b2'}], 'region_status': 'detected'},
    )
    assert resp.status_code == 200, resp.text
    stored = fake_os._docs['crop-1'][F.boxes]
    assert stored[0]['state'] == 'accepted'
    assert stored[1]['state'] == 'accepted'  # was proposed -> now accepted
    assert fake_os._docs['crop-1'][F.status] == 'detected'
    assert fake_os._docs['crop-1'][F.validated] is True


def test_put_regions_stale_revision_returns_409_region_conflict(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': 'b1'}], 'expected_region_revision': 999},
    )
    assert resp.status_code == 409, resp.text
    detail = resp.json()['detail']
    assert detail['error'] == 'region_conflict'
    assert detail['current_region_revision'] == 3
    assert set(detail['current_box_ids']) == {'b1', 'b2'}


def test_put_regions_matching_revision_succeeds(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': 'b1'}], 'expected_region_revision': 3},
    )
    assert resp.status_code == 200, resp.text


def test_put_regions_unknown_box_id_is_422(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': 'b99'}]},
    )
    assert resp.status_code == 422


def test_put_regions_over_max_boxes_per_write_is_422_too_many_boxes(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_REGION_MAX_BOXES_PER_WRITE', '1')
    import src.config.curation as curation_config_mod

    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    try:
        resp = app_client.put(
            '/curation/projects/default/crops/crop-1/regions',
            json={'boxes': [{'box_id': 'b1'}, {'box_id': 'b2'}]},
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()['detail']['error'] == 'too_many_boxes'
    finally:
        monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)


def test_put_regions_missing_crop_returns_404(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/does-not-exist/regions',
        json={'boxes': []},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT crops-batch_regions
# ---------------------------------------------------------------------------


def test_batch_regions_clears_every_crop(app_client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/batch_regions',
        json={'crop_ids': ['crop-1', 'crop-2'], 'boxes': []},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['updated'] == 2
    assert fake_os._docs['crop-1'][F.boxes] == []
    assert fake_os._docs['crop-2'][F.boxes] == []


def test_batch_regions_rejects_box_id_in_payload(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/batch_regions',
        json={'crop_ids': ['crop-1'], 'boxes': [{'box_id': 'b1'}]},
    )
    assert resp.status_code == 422
    assert resp.json()['detail']['error'] == 'box_id_in_batch'


# ---------------------------------------------------------------------------
# PATCH crops-id-regions-box_id
# ---------------------------------------------------------------------------


def test_patch_region_box_accept(app_client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/regions/b2',
        json={'state': 'accepted'},
    )
    assert resp.status_code == 200, resp.text
    stored = {b['box_id']: b for b in fake_os._docs['crop-1'][F.boxes]}
    assert stored['b2']['state'] == 'accepted'
    assert stored['b1']['state'] == 'accepted'  # untouched sibling


def test_patch_region_box_reject_does_not_touch_siblings(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/regions/b1',
        json={'state': 'rejected'},
    )
    assert resp.status_code == 200, resp.text
    stored = {b['box_id']: b for b in fake_os._docs['crop-1'][F.boxes]}
    assert stored['b1']['state'] == 'rejected'
    assert stored['b2']['state'] == 'proposed'


def test_patch_region_box_unknown_id_is_422(app_client: TestClient) -> None:
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/regions/b99',
        json={'state': 'accepted'},
    )
    assert resp.status_code == 422


def test_patch_region_box_requires_a_field(app_client: TestClient) -> None:
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/regions/b1',
        json={},
    )
    assert resp.status_code == 400


def test_patch_region_box_stale_revision_is_409(
    app_client: TestClient,
) -> None:
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/regions/b1',
        json={'state': 'rejected', 'expected_region_revision': 1},
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST regions-batch_box_state
# ---------------------------------------------------------------------------


def test_batch_box_state_flips_only_named_boxes(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.post(
        '/curation/projects/default/regions/batch_box_state',
        json={'targets': [{'crop_id': 'crop-1', 'box_id': 'b2'}], 'state': 'accepted'},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['updated'] == 1
    stored = {b['box_id']: b for b in fake_os._docs['crop-1'][F.boxes]}
    assert stored['b2']['state'] == 'accepted'
    assert stored['b1']['state'] == 'accepted'


def test_batch_box_state_unknown_box_id_is_invalid_not_updated(
    app_client: TestClient,
) -> None:
    resp = app_client.post(
        '/curation/projects/default/regions/batch_box_state',
        json={'targets': [{'crop_id': 'crop-1', 'box_id': 'b99'}], 'state': 'accepted'},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['updated'] == 0
    assert len(body['invalid']) == 1


# ---------------------------------------------------------------------------
# W8c: per-route `state` validation against BOX_STATE_ROUTES
# ---------------------------------------------------------------------------


def test_put_regions_unknown_state_is_422(app_client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': 'b1', 'state': 'confirmed'}]},
    )
    assert resp.status_code == 422
    assert fake_os._docs['crop-1'][F.boxes][0]['state'] == 'accepted', (
        'unchanged, rejected up front'
    )


def test_batch_regions_unknown_state_is_422(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/batch_regions',
        json={'crop_ids': ['crop-2'], 'boxes': [{'box_id': None, 'state': 'bogus'}]},
    )
    assert resp.status_code == 422


def test_patch_region_box_unknown_state_is_422(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/regions/b1',
        json={'state': 'bogus'},
    )
    assert resp.status_code == 422
    assert fake_os._docs['crop-1'][F.boxes][0]['state'] == 'accepted', (
        'unchanged, rejected up front'
    )


def test_batch_box_state_unknown_state_is_422(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.post(
        '/curation/projects/default/regions/batch_box_state',
        json={'targets': [{'crop_id': 'crop-1', 'box_id': 'b2'}], 'state': 'bogus'},
    )
    assert resp.status_code == 422
    assert fake_os._docs['crop-1'][F.boxes][1]['state'] == 'proposed', (
        'unchanged, rejected up front'
    )
