"""``on_negative_frame`` on the review queue: machine items an import found on
a reviewed-negative frame (likely false positives), served as a filter spec
and applied on every tab."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation import base_curation_config
from src.services.curation import review_queries


ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default/review'


def _item(crop_id: str, **over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'image_id': f'img-{crop_id}',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_id': 3,
        'class_name': 'car',
        'class_source': 'vlm_unmatched',
        'class_validated': False,
        'updated_at': '2026-10-01T00:00:00+00:00',
    }
    doc.update(over)
    return doc


SEEDED = {
    'on_negative': _item('on_negative', on_negative_frame=True),
    'elsewhere': _item('elsewhere', on_negative_frame=False),
    'unmarked': _item('unmarked'),
}


async def _noop(*_a: Any, **_k: Any) -> None:
    return None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', _noop)
    fake = QueryFakeOpenSearch({ITEMS: dict(SEEDED)})
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def _ids(client: TestClient, **params: Any) -> set[str]:
    r = client.get(f'{BASE}/all', params=params)
    assert r.status_code == 200, r.text
    return {i['crop_id'] for i in r.json()['items']}


def test_the_filter_selects_in_both_directions(client: TestClient) -> None:
    assert _ids(client) == {'on_negative', 'elsewhere', 'unmarked'}
    assert _ids(client, on_negative_frame=True) == {'on_negative'}
    assert _ids(client, on_negative_frame=False) == {'elsewhere', 'unmarked'}


def test_the_locate_route_honours_the_same_filter(client: TestClient) -> None:
    hit = client.get(
        f'{BASE}/all/locate', params={'crop_id': 'on_negative', 'on_negative_frame': True}
    )
    miss = client.get(
        f'{BASE}/all/locate', params={'crop_id': 'elsewhere', 'on_negative_frame': True}
    )

    assert hit.json()['in_queue'] is True
    assert miss.json()['in_queue'] is False


def test_the_catalog_serves_the_spec_on_every_tab_that_honours_it(client: TestClient) -> None:
    tabs = client.get(f'{BASE}/tabs').json()['tabs']

    assert tabs
    for tab in tabs:
        assert 'on_negative_frame' in tab['filters']
        spec = next(s for s in tab['filter_specs'] if s['param'] == 'on_negative_frame')
        assert [o['value'] for o in spec['options']] == [
            '',
            'true',
            'false',
        ]  # '' = Any: omit the param
    assert 'on_negative_frame' in review_queries.COMMON_FILTERS
