"""`POST /images/{image_id}/items`: a person draws a box and it becomes an item.

The item carries a human class and human geometry at once, the class is resolved by name, a retry
of the same `request_id` returns the first item, a box on a held-out image is held out, and an
embedding failure keeps the write and says so. Every write runs through the real router and
service against the query-evaluating in-memory OpenSearch.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.reprocess_fixtures import FakePE, jpeg_bytes, make_fake, servable_root
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.services.classes.registry import get_class_registry


if TYPE_CHECKING:
    from pathlib import Path

    from curation.query_fakes import QueryFakeOpenSearch

CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)
BASE = '/curation/projects/default'
BOX = [0.2, 0.2, 0.6, 0.7]


class _BrokenPE(FakePE):
    async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> Any:
        raise RuntimeError('encoder down')


def _client(fake: Any, pe: Any) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    app.state.pe_encoder = pe
    return TestClient(app)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[QueryFakeOpenSearch, str]:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    get_class_registry().seed(['car', 'truck'])  # type: ignore[attr-defined]
    get_class_registry().deprecate(1)  # type: ignore[attr-defined]
    fake = make_fake(
        [{'crop_id': 'det-1', 'image_id': 'img-1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2]}],
        [
            {
                'image_id': 'img-1',
                'image_path': str(path),
                'width': 400,
                'height': 300,
                'source': 'disk',
            }
        ],
    )
    return fake, 'img-1'


def _body(**over: Any) -> dict[str, Any]:
    return {'bbox_norm': BOX, 'class_name': 'car', 'request_id': str(uuid.uuid4()), **over}


def _post(client: TestClient, body: dict[str, Any], image_id: str = 'img-1') -> Any:
    return client.post(f'{BASE}/images/{image_id}/items', json=body)


def test_drawn_item_has_human_class_and_geometry(world: Any) -> None:
    fake, image_id = world
    resp = _post(_client(fake, FakePE()), _body())
    assert resp.status_code == 201, resp.text
    out = resp.json()
    assert out['created'] is True
    doc = fake.docs(ITEMS)[out['item']['crop_id']]
    assert (doc['class_source'], doc['label_source'], doc['class_validated']) == (
        'human',
        'human',
        True,
    )
    assert (doc['geometry_source'], doc['geometry_revision']) == ('human', 1)
    assert doc['bbox_norm'] == BOX
    assert doc['crop_area_norm'] == pytest.approx(0.2)
    assert doc['crop_rank_in_image'] == 1  # larger than the detector box on the image
    assert fake.docs(ITEMS)['det-1']['crop_rank_in_image'] == 2
    assert doc['image_id'] == image_id


def test_class_is_resolved_by_name_and_deprecated_is_refused(world: Any) -> None:
    fake, _ = world
    client = _client(fake, FakePE())
    assert _post(client, _body(class_name='  CAR ')).status_code == 201
    for name in ('truck', 'nonsense'):  # truck is deprecated
        resp = _post(client, _body(class_name=name))
        assert resp.status_code == 422
        assert resp.json()['detail']['error'] == 'unknown_class'
        assert resp.json()['detail']['class_name'] == name


def test_repeat_with_the_same_request_id_returns_the_first_item(world: Any) -> None:
    fake, _ = world
    client = _client(fake, FakePE())
    body = _body()
    first, second = _post(client, body), _post(client, body)
    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()['created'] is False
    assert second.json()['item']['crop_id'] == first.json()['item']['crop_id']
    assert len(fake.docs(ITEMS)) == 2


@pytest.mark.parametrize(
    ('box', 'code'),
    [
        ([0.2, 0.2, 1.2, 0.7], 'box_out_of_image'),
        ([-0.1, 0.2, 0.5, 0.7], 'box_out_of_image'),
        ([0.5, 0.2, 0.2, 0.7], 'degenerate_box'),
        ([0.2, 0.2, 0.2, 0.7], 'degenerate_box'),
        ([0.2, 0.2, 0.202, 0.7], 'box_too_small'),
    ],
)
def test_box_refusals(world: Any, box: list[float], code: str) -> None:
    fake, _ = world
    resp = _post(_client(fake, FakePE()), _body(bbox_norm=box))
    assert resp.status_code == 422
    detail = resp.json()['detail']
    assert detail['error'] == code
    if code == 'box_too_small':
        assert detail['min_side'] == 0.004
    assert len(fake.docs(ITEMS)) == 1


def test_foreign_image_is_404(world: Any) -> None:
    fake, _ = world
    resp = _post(_client(fake, FakePE()), _body(), image_id='other-projects-image')
    assert resp.status_code == 404
    assert resp.json()['detail']['error'] == 'image_not_found'
    assert len(fake.docs(ITEMS)) == 1


def test_region_seed_set_when_the_class_is_a_region_parent(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import RegionStatus, get_region_fields
    from src.services.curation import image_item_write

    monkeypatch.setattr(
        image_item_write, 'region_seed_status', lambda _item: RegionStatus.PENDING_DETECTION
    )
    fake, _ = world
    out = _post(_client(fake, FakePE()), _body()).json()
    assert (
        fake.docs(ITEMS)[out['item']['crop_id']][get_region_fields().status] == 'pending_detection'
    )


def test_new_item_on_a_held_out_image_is_held_out(world: Any) -> None:
    fake, _ = world
    fake.docs(ITEMS)['det-1'].update({'test_holdout': True, 'dataset_split': 'test'})
    out = _post(_client(fake, FakePE()), _body()).json()
    doc = fake.docs(ITEMS)[out['item']['crop_id']]
    assert (doc['test_holdout'], doc['dataset_split']) == (True, 'test')


def test_a_new_item_on_an_open_image_is_not_held_out(world: Any) -> None:
    fake, _ = world
    out = _post(_client(fake, FakePE()), _body()).json()
    assert fake.docs(ITEMS)[out['item']['crop_id']]['test_holdout'] is False


def test_embedded_on_success(world: Any) -> None:
    fake, _ = world
    out = _post(_client(fake, FakePE()), _body()).json()
    assert (out['embedding_state'], out['embedding_message']) == ('embedded', None)
    assert fake.docs(ITEMS)[out['item']['crop_id']]['pe_embedding'] == [0.0, 0.0, 1.0]


def test_embed_failure_keeps_the_item_deferred(world: Any) -> None:
    fake, _ = world
    resp = _post(_client(fake, _BrokenPE()), _body())
    assert resp.status_code == 201
    out = resp.json()
    assert out['embedding_state'] == 'deferred'
    assert 'Embed missing items' in out['embedding_message']
    assert 'pe_embedding' not in fake.docs(ITEMS)[out['item']['crop_id']]


def test_near_dup_pending_marked_added(world: Any, pending_calls: Any) -> None:
    fake, _ = world
    out = _post(_client(fake, FakePE()), _body()).json()
    crop_id = out['item']['crop_id']
    assert ('item', [crop_id], 'added') in pending_calls.calls


def test_response_carries_the_image_boxes_and_counts(world: Any) -> None:
    fake, _ = world
    out = _post(_client(fake, FakePE()), _body()).json()
    assert {b['crop_id'] for b in out['image_boxes']} == {'det-1', out['item']['crop_id']}
    assert out['image_counts']['items'] == 2
    assert out['image_counts']['validated'] == 1
