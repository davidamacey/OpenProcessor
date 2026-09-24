"""Region boxes in the item-crop ("parent") frame are converted by the server.

A client draws a region on the item crop. It sends that box with
``frame='parent'`` and the server projects it into the source frame
through the item's own stored ``bbox_norm``. Every item also carries
``region_bbox_in_parent``, the stored region re-expressed in the crop
frame, so a client never does frame math in either direction.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_regions_router import _FakeRegionOS
from src.config import get_region_fields
from src.services.curation.wire import serialize_item


F = get_region_fields()
PARENT = [0.2, 0.4, 0.6, 0.8]  # item box in the source frame


@pytest.fixture
def fake_os() -> _FakeRegionOS:
    return _FakeRegionOS(
        {
            'c1': {'crop_id': 'c1', 'bbox_norm': list(PARENT)},
            'c2': {'crop_id': 'c2', 'bbox_norm': list(PARENT)},
            'nobox': {'crop_id': 'nobox'},
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


def test_put_region_parent_frame_is_projected_to_source(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = client.put(
        '/curation/crops/c1/region',
        json={'region_bbox_norm': [0.5, 0.5, 1.0, 1.0], 'frame': 'parent'},
    )
    assert resp.status_code == 200, resp.text
    stored = fake_os._docs['c1'][F.bbox_norm]
    assert stored == pytest.approx([0.4, 0.6, 0.6, 0.8])
    assert fake_os._docs['c1'][F.bbox_frame] == 'source'
    item = resp.json()['item']
    assert item['region_bbox_norm'] == pytest.approx([0.4, 0.6, 0.6, 0.8])
    assert item['region_bbox_in_parent'] == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_put_region_default_frame_is_source(client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = client.put('/curation/crops/c1/region', json={'region_bbox_norm': [0.3, 0.5, 0.4, 0.6]})
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['c1'][F.bbox_norm] == [0.3, 0.5, 0.4, 0.6]


def test_put_region_parent_frame_without_item_box_is_422(client: TestClient) -> None:
    resp = client.put(
        '/curation/crops/nobox/region',
        json={'region_bbox_norm': [0.1, 0.1, 0.5, 0.5], 'frame': 'parent'},
    )
    assert resp.status_code == 422, resp.text


def test_put_region_unknown_frame_is_422(client: TestClient) -> None:
    resp = client.put(
        '/curation/crops/c1/region',
        json={'region_bbox_norm': [0.1, 0.1, 0.5, 0.5], 'frame': 'crop'},
    )
    assert resp.status_code == 422


def test_batch_region_parent_frame_uses_each_items_own_box(
    client: TestClient, fake_os: _FakeRegionOS
) -> None:
    fake_os._docs['c2']['bbox_norm'] = [0.0, 0.0, 0.5, 0.5]
    resp = client.put(
        '/curation/crops/batch_region',
        json={
            'crop_ids': ['c1', 'c2', 'nobox'],
            'region_bbox_norm': [0.0, 0.0, 0.5, 0.5],
            'frame': 'parent',
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['updated'] == 2
    assert [i['crop_id'] for i in body['invalid']] == ['nobox']
    assert fake_os._docs['c1'][F.bbox_norm] == pytest.approx([0.2, 0.4, 0.4, 0.6])
    assert fake_os._docs['c2'][F.bbox_norm] == pytest.approx([0.0, 0.0, 0.25, 0.25])


def test_wire_region_bbox_in_parent() -> None:
    item = serialize_item(
        {'bbox_norm': list(PARENT), F.bbox_norm: [0.4, 0.6, 0.6, 0.8]}, 'x', api_prefix=''
    )
    assert item['region_bbox_in_parent'] == pytest.approx([0.5, 0.5, 1.0, 1.0])


@pytest.mark.parametrize(
    'doc',
    [
        {'bbox_norm': list(PARENT)},  # no region
        {F.bbox_norm: [0.4, 0.6, 0.6, 0.8]},  # no item box
        {'bbox_norm': [0.5, 0.5, 0.5, 0.9], F.bbox_norm: [0.4, 0.6, 0.6, 0.8]},  # degenerate
    ],
)
def test_wire_region_bbox_in_parent_null_when_not_derivable(doc: dict[str, Any]) -> None:
    assert serialize_item(doc, 'x', api_prefix='')['region_bbox_in_parent'] is None


def test_wire_region_bbox_in_parent_reads_crop_frame_verbatim() -> None:
    item = serialize_item(
        {'bbox_norm': list(PARENT), F.bbox_norm: [0.1, 0.1, 0.2, 0.2], F.bbox_frame: 'crop'},
        'x',
        api_prefix='',
    )
    assert item['region_bbox_in_parent'] == [0.1, 0.1, 0.2, 0.2]
