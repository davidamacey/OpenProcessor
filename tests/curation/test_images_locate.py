"""`GET /images/locate`: where an image sits in the gallery order, and the cursors around it."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from curation.paging_fakes import AggregatingPagingOpenSearch, images_client
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config


CFG = base_curation_config()
IMAGES = index_name(CFG, IndexRole.IMAGES)
ITEMS = CFG.items_index
BASE = '/curation/projects/default'
N = 30


def _docs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    images, items = {}, {}
    for i in range(N):
        images[f'img{i:03d}'] = {
            'image_id': f'img{i:03d}',
            'image_path': f'/data/{i:03d}.jpg',
            'width': 640,
            'height': 480,
            'source': 'disk' if i % 2 else 'cam',
            # ties on purpose: the image_id tiebreak has to place equal values
            'indexed_at': f'2026-01-{(i % 6) + 1:02d}T00:00:00Z',
        }
        items[f'c{i:03d}'] = {
            'crop_id': f'c{i:03d}',
            'image_id': f'img{i:03d}',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_name': 'car' if i % 3 else 'bus',
        }
    return images, items


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    images, items = _docs()
    return images_client(AggregatingPagingOpenSearch({IMAGES: images, ITEMS: items}), monkeypatch)


def _ids(client: TestClient, **params: Any) -> list[str]:
    r = client.get(f'{BASE}/images', params=params)
    assert r.status_code == 200, r.text
    return [c['image_id'] for c in r.json()['items']]


@pytest.mark.parametrize('sort', ['indexed_at:asc', 'indexed_at:desc', 'file_name:asc'])
def test_index_and_cursors_walk_to_the_image(client: TestClient, sort: str) -> None:
    everything = _ids(client, sort=sort, limit=200)
    for target in ('img000', 'img013', 'img025', everything[-1]):
        r = client.get(f'{BASE}/images/locate', params={'target_image_id': target, 'sort': sort})
        assert r.status_code == 200, r.text
        got = r.json()
        position = everything.index(target)
        assert got['in_view'] is True
        assert got['index'] == position
        assert got['total'] == N
        assert got['reason'] is None
        if position == 0:
            assert got['cursor_before'] is None
        else:
            page = _ids(client, sort=sort, limit=5, cursor=got['cursor_before'])
            assert page == everything[position : position + 5]
        after = _ids(client, sort=sort, limit=5, cursor=got['cursor_after'])
        assert after == everything[position + 1 : position + 6]


def test_filtered_out_image_says_so(client: TestClient) -> None:
    r = client.get(f'{BASE}/images/locate', params={'target_image_id': 'img001', 'source': 'cam'})
    assert r.status_code == 200, r.text
    got = r.json()
    assert (got['in_view'], got['index'], got['reason']) == (False, None, 'filtered_out')
    assert got['cursor_before'] is None
    assert got['cursor_after'] is None
    assert got['reason_message']


def test_an_unknown_image_is_not_found(client: TestClient) -> None:
    got = client.get(f'{BASE}/images/locate', params={'target_image_id': 'nope'}).json()
    assert (got['in_view'], got['reason']) == (False, 'not_found')


def test_a_bad_sort_is_refused(client: TestClient) -> None:
    r = client.get(f'{BASE}/images/locate', params={'target_image_id': 'img001', 'sort': 'bogus'})
    assert r.status_code == 400
