"""The Table's "Unlabeled" column sorts through the served `unlabeled_count` sort (R-13).

It is sortable only while every image has its stored counts; before that the column is not
sortable, the option is not served and asking for it is 422 `image_counts_not_ready`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient
from fixtures.fake_valkey import FakeValkeySets

from curation.paging_fakes import images_client
from curation.query_fakes import QueryFakeOpenSearch
from src.clients.image_counts_dirty import DirtyImages
from src.config.curation_accessors import base_curation_config
from src.services.curation import image_counts_refresh as refresh


CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = CFG.images_index
BASE = '/curation/projects/default'


def _world() -> QueryFakeOpenSearch:
    images = {f'img-{n}': {'image_id': f'img-{n}', 'image_path': f'/d/{n}.jpg'} for n in 'abcd'}
    items: dict[str, dict[str, Any]] = {}
    # img-a: 3 unlabeled, img-b: 1 unlabeled, img-c: 0 unlabeled, img-d: no items
    for i in range(3):
        items[f'a{i}'] = {'crop_id': f'a{i}', 'image_id': 'img-a'}
    items['b0'] = {'crop_id': 'b0', 'image_id': 'img-b'}
    items['b1'] = {'crop_id': 'b1', 'image_id': 'img-b', 'class_id': 1}
    items['c0'] = {'crop_id': 'c0', 'image_id': 'img-c', 'class_id': 1}
    return QueryFakeOpenSearch({IMAGES: images, ITEMS: items})


def _client(counted: bool, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake = _world()
    if counted:
        asyncio.run(refresh.drain(fake, DirtyImages(FakeValkeySets())))
    return images_client(fake, monkeypatch)


def _unlabeled(columns: list[dict[str, Any]]) -> dict[str, Any]:
    return next(c for c in columns if c['id'] == 'unlabeled')


def test_the_column_is_not_sortable_before_the_counts_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(False, monkeypatch)
    page = client.get(
        f'{BASE}/table', params={'rows': 'images', 'columns': 'file,unlabeled'}
    ).json()
    assert _unlabeled(page['columns'])['sortable'] is False
    assert not [o for o in page['sort_options'] if o['id'].startswith('unlabeled')]
    listed = client.get(f'{BASE}/table/columns', params={'rows': 'images'}).json()
    assert _unlabeled(listed['columns'])['sortable'] is False


def test_asking_for_the_sort_before_the_counts_exist_is_refused_with_a_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(False, monkeypatch)
    out = client.get(f'{BASE}/table', params={'rows': 'images', 'sort': 'unlabeled:desc'})
    assert out.status_code == 422
    assert out.json()['detail']['error'] == 'image_counts_not_ready'


@pytest.mark.parametrize(('direction', 'first'), [('desc', 'img-a'), ('asc', 'img-c')])
def test_the_column_sorts_by_the_stored_count_once_counted(
    monkeypatch: pytest.MonkeyPatch, direction: str, first: str
) -> None:
    client = _client(True, monkeypatch)
    params: dict[str, str | int] = {
        'rows': 'images',
        'columns': 'file,unlabeled',
        'sort': f'unlabeled:{direction}',
        'limit': 200,
    }
    page = client.get(f'{BASE}/table', params=params).json()
    assert _unlabeled(page['columns'])['sortable'] is True
    assert f'unlabeled:{direction}' in [o['id'] for o in page['sort_options']]
    assert page['sort_applied'] == f'unlabeled:{direction}'
    served = [r['image_id'] for r in page['items']]
    assert served[0] == first
    values = [r['cells']['unlabeled']['value'] for r in page['items']]
    assert values == sorted(values, reverse=direction == 'desc')
    listed = client.get(
        f'{BASE}/images', params={'sort': f'unlabeled_count:{direction}', 'limit': 200}
    ).json()
    assert served == [c['image_id'] for c in listed['items']]
    columns = client.get(f'{BASE}/table/columns', params={'rows': 'images'}).json()['columns']
    assert _unlabeled(columns)['sortable'] is True
