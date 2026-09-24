"""Region-status invariants are enforced by the backend, not the client.

Every human region writer (``PUT /crops/{id}/region``, ``PUT
/crops/batch_region``, ``PATCH /crops/{id}/region_meta``, ``POST
/regions/batch_status``) must leave the stored region self-consistent:

- a status whose lifecycle entry says ``clears_box`` (``no_region_visible``)
  removes the box and score, whichever writer set it;
- ``region_verified`` is derived from the status (``detected`` -> true,
  anything else -> false), never taken from the request;
- ``detected`` without a box is refused;
- each writer returns the post-write item(s) in the shared wire format.

``GET /regions/statuses`` serves the lifecycle vocabulary those rules
come from.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_regions_router import _FakeRegionOS
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.routers.curation._common import HUMAN_REGION_STATUS_VALUES
from src.services.curation.wire import ITEM_WIRE_KEYS


F = get_region_fields()
BOX = [0.1, 0.2, 0.3, 0.4]


def _boxed(crop_id: str) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'bbox_norm': [0.0, 0.0, 0.5, 0.5],
        F.bbox_norm: list(BOX),
        F.score: 0.91,
        F.status: RegionStatus.DETECTED.value,
        F.verified: True,
    }


@pytest.fixture
def fake_os() -> _FakeRegionOS:
    return _FakeRegionOS(
        {
            'boxed-1': _boxed('boxed-1'),
            'boxed-2': _boxed('boxed-2'),
            'empty-1': {'crop_id': 'empty-1', F.status: 'pending_detection'},
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


# ---------------------------------------------------------------- box clearing


def test_batch_status_no_region_visible_clears_box(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['boxed-1', 'boxed-2'], 'region_status': 'no_region_visible'},
    )
    assert resp.status_code == 200, resp.text
    for cid in ('boxed-1', 'boxed-2'):
        doc = fake_os._docs[cid]
        assert doc[F.bbox_norm] is None
        assert doc[F.score] is None
        assert doc[F.verified] is False


def test_patch_meta_no_region_visible_clears_box(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = client.patch(
        '/curation/crops/boxed-1/region_meta', json={'region_status': 'no_region_visible'}
    )
    assert resp.status_code == 200, resp.text
    doc = fake_os._docs['boxed-1']
    assert doc[F.bbox_norm] is None
    assert doc[F.score] is None
    assert doc[F.verified] is False


def test_false_positive_keeps_box(client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = client.patch(
        '/curation/crops/boxed-1/region_meta', json={'region_status': 'false_positive'}
    )
    assert resp.status_code == 200, resp.text
    doc = fake_os._docs['boxed-1']
    assert doc[F.bbox_norm] == BOX
    assert doc[F.verified] is False


# ------------------------------------------------------------ derived verified


def test_batch_status_detected_derives_verified_ignoring_client_value(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    fake_os._docs['boxed-1'][F.verified] = False
    resp = client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['boxed-1'], 'region_status': 'detected', 'region_verified': False},
    )
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['boxed-1'][F.verified] is True


def test_batch_status_rejection_unsets_verified(client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['boxed-1'], 'region_status': 'verify_rejected', 'region_verified': True},
    )
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['boxed-1'][F.verified] is False
    assert fake_os._docs['boxed-1'][F.bbox_norm] == BOX


def test_patch_meta_detected_sets_verified(client: TestClient, fake_os: _FakeRegionOS) -> None:
    fake_os._docs['boxed-1'][F.verified] = False
    resp = client.patch('/curation/crops/boxed-1/region_meta', json={'region_status': 'detected'})
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['boxed-1'][F.verified] is True


def test_patch_meta_detected_without_box_is_refused(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = client.patch('/curation/crops/empty-1/region_meta', json={'region_status': 'detected'})
    assert resp.status_code == 422, resp.text
    assert fake_os._docs['empty-1'][F.status] == 'pending_detection'


def test_batch_status_detected_without_box_is_reported_invalid(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['boxed-1', 'empty-1'], 'region_status': 'detected'},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['updated'] == 1
    assert [i['crop_id'] for i in body['invalid']] == ['empty-1']
    assert fake_os._docs['empty-1'][F.status] == 'pending_detection'


# ------------------------------------------------------------ post-write items


def test_patch_meta_returns_post_write_item(client: TestClient) -> None:
    resp = client.patch(
        '/curation/crops/boxed-1/region_meta', json={'region_status': 'no_region_visible'}
    )
    item = resp.json()['item']
    assert set(item) == ITEM_WIRE_KEYS
    assert item['region_status'] == 'no_region_visible'
    assert item['region_bbox_norm'] is None
    assert item['region_verified'] is False
    assert item['region_validated'] is True


def test_batch_status_returns_post_write_items(client: TestClient) -> None:
    resp = client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['boxed-1', 'boxed-2'], 'region_status': 'false_positive'},
    )
    items = resp.json()['items']
    assert [i['crop_id'] for i in items] == ['boxed-1', 'boxed-2']
    assert all(set(i) == ITEM_WIRE_KEYS for i in items)
    assert all(i['region_status'] == 'false_positive' for i in items)


def test_put_region_returns_post_write_item(client: TestClient) -> None:
    resp = client.put('/curation/crops/empty-1/region', json={'region_bbox_norm': BOX})
    assert resp.status_code == 200, resp.text
    item = resp.json()['item']
    assert set(item) == ITEM_WIRE_KEYS
    assert item['region_status'] == 'detected'
    assert item['region_verified'] is True
    assert item['region_bbox_norm'] == BOX


def test_put_region_null_unsets_verified(client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = client.put('/curation/crops/boxed-1/region', json={'region_bbox_norm': None})
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['boxed-1'][F.verified] is False
    assert resp.json()['item']['region_status'] == 'no_region_visible'


def test_batch_region_returns_post_write_items(client: TestClient) -> None:
    resp = client.put(
        '/curation/crops/batch_region',
        json={'crop_ids': ['boxed-1', 'boxed-2'], 'region_bbox_norm': None},
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()['items']
    assert [i['crop_id'] for i in items] == ['boxed-1', 'boxed-2']
    assert all(i['region_bbox_norm'] is None for i in items)


# ------------------------------------------------------------- the vocabulary


def test_status_catalog_covers_every_status(client: TestClient) -> None:
    resp = client.get('/curation/regions/statuses')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    values = [s['value'] for s in body['statuses']]
    assert values == [s.value for s in RegionStatus]
    writable = {s['value'] for s in body['statuses'] if s['human_writable']}
    assert writable == {s.value for s in HUMAN_REGION_STATUS_VALUES}
    assert body['confirm_status'] == 'detected'
    assert body['reject_status'] == 'no_region_visible'
    assert body['false_positive_status'] == 'false_positive'
    by_value = {s['value']: s for s in body['statuses']}
    assert by_value['no_region_visible']['clears_box'] is True
    assert by_value['false_positive']['clears_box'] is False
    assert by_value['pending_detection']['role'] == 'pending'
    assert by_value['pending_detection']['terminal'] is False
    assert by_value['detected']['role'] == 'positive'
    assert {k for k, v in by_value.items() if v['wants_reason']} == {
        'verify_rejected',
        'no_region_visible',
    }
    for entry in body['statuses']:
        assert set(entry) == {
            'value',
            'label',
            'role',
            'terminal',
            'human_writable',
            'clears_box',
            'wants_reason',
        }
