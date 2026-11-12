"""`POST /selection/summary`: what a bulk action would touch, counted in one aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.routers.curation import selection_summary as route


if TYPE_CHECKING:
    import pytest

CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)
URL = '/curation/projects/default/selection/summary'


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    async def _policy() -> str:
        return 'store'

    monkeypatch.setattr(route, 'get_location_policy', _policy)
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _item(crop_id: str, image: str, **extra: Any) -> dict[str, Any]:
    return {'crop_id': crop_id, 'image_id': image, 'bbox_norm': [0, 0, 1, 1], **extra}


def _world() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            ITEMS: {
                'a1': _item('a1', 'img-a', class_id=1, class_name='car', cluster_uid='u1'),
                'a2': _item(
                    'a2',
                    'img-a',
                    class_id=1,
                    class_name='car',
                    class_source='human',
                    cluster_uid='u1',
                ),
                'b1': _item('b1', 'img-b', class_id=2, class_name='bus', cluster_uid='u2'),
                'b2': _item('b2', 'img-b', cluster_uid='u2'),
                'c1': _item('c1', 'img-c', class_id=1, class_name='car', class_excluded=True),
            },
            IMAGES: {
                'img-a': {'image_id': 'img-a', 'capture.location': {'lat': 1, 'lon': 2}},
                'img-b': {'image_id': 'img-b'},
                'img-c': {'image_id': 'img-c'},
            },
        }
    )


def test_counts_equal_crops_with_the_same_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)
    body = client.post(URL, json={'filter': {'class_names': ['car']}}).json()
    assert (body['detections'], body['images']) == (2, 1)
    assert body['by_class'] == [{'class_name': 'car', 'count': 2}]
    assert body['located'] == 1
    assert body['join_truncated'] is False


def test_image_selection_expands_to_matching_items(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)
    body = client.post(URL, json={'filter': {'image_ids': ['img-a', 'img-b']}}).json()
    assert (body['detections'], body['images'], body['clusters']) == (4, 2, 2)
    assert {c['class_name']: c['count'] for c in body['by_class']} == {
        'car': 2,
        'bus': 1,
        'unlabeled': 1,
    }
    assert body['summary_text'].startswith('4 detections in 2 images')


def test_locked_counts_class_box_and_geometry_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)
    body = client.post(URL, json={'crop_ids': ['a1', 'a2', 'b1']}).json()
    assert body['locked'] == 1
    assert body['summary_text'] == '3 detections in 2 images · 1 locked, skipped'


def test_locked_counts_a_human_box_and_a_human_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'g1': _item('g1', 'img-a', geometry_source='human'),
                'g2': _item(
                    'g2',
                    'img-a',
                    class_excluded=True,
                    excluded_by='human',
                    excluded_reason='not_an_object',
                ),
                'g3': _item('g3', 'img-a'),
            },
            IMAGES: {'img-a': {'image_id': 'img-a'}},
        }
    )
    client = _client(fake, monkeypatch)
    body = client.post(URL, json={'crop_ids': ['g1', 'g2', 'g3']}).json()
    assert body['locked'] == 2


def test_over_the_bulk_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.services.curation.selection_summary.MAX_BULK_SELECTION', 2)
    client = _client(_world(), monkeypatch)
    response = client.post(URL, json={'filter': {'image_ids': ['img-a', 'img-b']}})
    assert response.status_code == 422


def test_a_location_filter_resolves_the_image_join(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)
    body = client.post(URL, json={'filter': {'has_location': True}}).json()
    assert (body['detections'], body['images']) == (2, 1)
    assert body['located'] == 1


def test_a_geo_box_filter_resolves_the_image_join(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)
    box = {'west': 0.0, 'south': 0.0, 'east': 5.0, 'north': 5.0}
    body = client.post(URL, json={'filter': {'geo_bbox': box}}).json()
    assert (body['detections'], body['images']) == (2, 1)
