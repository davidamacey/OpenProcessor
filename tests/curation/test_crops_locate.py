"""`GET /crops/locate`: where a crop sits in any `/crops` order, and the cursors around it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.paging_fakes import PagingOpenSearch, crops_client
from src.config.curation_accessors import base_curation_config


if TYPE_CHECKING:
    from fastapi.testclient import TestClient

ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default'
N = 40


def _docs() -> dict[str, dict[str, Any]]:
    docs = {}
    for i in range(N):
        doc: dict[str, Any] = {
            'crop_id': f'c{i:03d}',
            'image_id': f'img-{i}',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            # ties on purpose: the crop_id tiebreak has to place equal values
            'confidence': round((i % 7) / 7, 4),
        }
        if i % 5:
            doc['class_name'] = 'car' if i % 2 else 'bus'
        docs[doc['crop_id']] = doc
    return docs


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return crops_client(PagingOpenSearch({ITEMS: _docs()}), monkeypatch)


def _ids(client: TestClient, **params: Any) -> list[str]:
    r = client.get(f'{BASE}/crops', params=params)
    assert r.status_code == 200, r.text
    return [c['id'] for c in r.json()['crops']]


@pytest.mark.parametrize('sort', ['confidence:asc', 'confidence:desc', 'class_name:asc'])
def test_index_and_cursors(client: TestClient, sort: str) -> None:
    everything = _ids(client, sort=sort, limit=500)
    for target in ('c000', 'c013', 'c025', everything[-1]):
        r = client.get(f'{BASE}/crops/locate', params={'crop_id': target, 'sort': sort})
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


def test_filtered_out(client: TestClient) -> None:
    r = client.get(f'{BASE}/crops/locate', params={'crop_id': 'c005', 'class_name': 'car'})
    assert r.status_code == 200, r.text
    got = r.json()
    assert (got['in_view'], got['index'], got['reason']) == (False, None, 'filtered_out')
    assert got['cursor_before'] is None
    assert got['cursor_after'] is None
    assert got['reason_message']


def test_an_unknown_crop_and_a_computed_order(client: TestClient) -> None:
    missing = client.get(f'{BASE}/crops/locate', params={'crop_id': 'nope'}).json()
    assert (missing['in_view'], missing['reason']) == (False, 'not_found')
    r = client.get(f'{BASE}/crops/locate', params={'crop_id': 'c001', 'order': 'outliers'})
    assert r.status_code == 400
