"""Region boxes in the item-crop ("parent") frame are converted by the server.

A client draws a region on the item crop. It sends that box to ``PUT
/crops/{id}/regions`` with ``frame='parent'`` and the server projects it
into the source frame through the item's own stored ``bbox_norm`` (not
through any region box). Every served box carries ``bbox_in_parent``, the
stored region re-expressed in the crop frame, so a client never does frame
math in either direction.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_regions_router import _FakeRegionOS
from src.config import get_region_fields
from src.services.curation.wire import serialize_item


# No-profile gating contract: this file exercises region routes, which
# require an active region profile (409 otherwise).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


F = get_region_fields()
PARENT = [0.2, 0.4, 0.6, 0.8]  # item box in the source frame


@pytest.fixture
def fake_os() -> _FakeRegionOS:
    return _FakeRegionOS(
        {
            'c1': {'crop_id': 'c1', 'bbox_norm': list(PARENT)},
            'nobox': {'crop_id': 'nobox'},
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


def _put(client: TestClient, crop_id: str, box: list[float], **body: Any) -> Any:
    return client.put(
        f'/curation/projects/default/crops/{crop_id}/regions',
        json={'boxes': [{'box_id': None, 'bbox_norm': box}], **body},
    )


def test_put_regions_parent_frame_is_projected_to_source(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = _put(client, 'c1', [0.5, 0.5, 1.0, 1.0], frame='parent')
    assert resp.status_code == 200, resp.text
    (stored,) = fake_os._docs['c1'][F.boxes]
    assert stored['bbox_norm'] == pytest.approx([0.4, 0.6, 0.6, 0.8])
    (wire_box,) = resp.json()['item']['region_boxes']
    assert wire_box['bbox_norm'] == pytest.approx([0.4, 0.6, 0.6, 0.8])
    assert wire_box['bbox_in_parent'] == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_put_regions_projects_through_the_item_box_not_a_region_box(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    """The frame a parent-frame box is drawn in is the *item crop*; an
    item that already holds a (different) region box must not change it."""
    fake_os._docs['c1'][F.boxes] = [
        {'box_id': 'b1', 'bbox_norm': [0.9, 0.9, 0.95, 0.95], 'state': 'accepted'}
    ]
    resp = client.put(
        '/curation/projects/default/crops/c1/regions',
        json={
            'boxes': [{'box_id': 'b1'}, {'box_id': None, 'bbox_norm': [0.5, 0.5, 1.0, 1.0]}],
            'frame': 'parent',
        },
    )
    assert resp.status_code == 200, resp.text
    stored = fake_os._docs['c1'][F.boxes]
    assert stored[1]['bbox_norm'] == pytest.approx([0.4, 0.6, 0.6, 0.8])


def test_put_regions_default_frame_is_source(client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = _put(client, 'c1', [0.3, 0.5, 0.4, 0.6])
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['c1'][F.boxes][0]['bbox_norm'] == [0.3, 0.5, 0.4, 0.6]


def test_put_regions_parent_frame_without_item_box_is_422(client: TestClient) -> None:
    resp = _put(client, 'nobox', [0.1, 0.1, 0.5, 0.5], frame='parent')
    assert resp.status_code == 422, resp.text


def test_put_regions_unknown_frame_is_422(client: TestClient) -> None:
    resp = _put(client, 'c1', [0.1, 0.1, 0.5, 0.5], frame='crop')
    assert resp.status_code == 422


def _wire_box(src_box: list[float], item_box: list[float] | None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        F.boxes: [{'box_id': 'b1', 'bbox_norm': src_box, 'state': 'accepted'}],
    }
    if item_box is not None:
        doc['bbox_norm'] = item_box
    (box,) = serialize_item(doc, 'x', api_prefix='')['region_boxes']
    return box


def test_wire_box_bbox_in_parent() -> None:
    box = _wire_box([0.4, 0.6, 0.6, 0.8], list(PARENT))
    assert box['bbox_in_parent'] == pytest.approx([0.5, 0.5, 1.0, 1.0])


@pytest.mark.parametrize(
    'item_box',
    [None, [0.5, 0.5, 0.5, 0.9]],  # no item box; degenerate item box
)
def test_wire_box_bbox_in_parent_null_when_not_derivable(item_box: list[float] | None) -> None:
    assert _wire_box([0.4, 0.6, 0.6, 0.8], item_box)['bbox_in_parent'] is None
