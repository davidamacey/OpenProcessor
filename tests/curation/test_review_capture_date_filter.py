"""The capture-date range on the review queue: every tab serves ``taken_from`` /
``taken_to`` as ``date`` specs, and the same params narrow ``GET /review/{tab}``."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation import review_tabs
from src.services.curation.review_filter_specs import FILTER_SPECS


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
    'march': _item('march', capture_taken_utc='2024-03-05T10:00:00'),
    'june': _item('june', capture_taken_utc='2024-06-20T10:00:00'),
    'undated': _item('undated'),
}


async def _noop(*_a: Any, **_k: Any) -> None:
    return None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.services.curation.index_bootstrap.ensure_indexes', _noop)
    fake = QueryFakeOpenSearch({ITEMS: dict(SEEDED)})
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def _ids(client: TestClient, **params: Any) -> set[str]:
    r = client.get(f'{BASE}/check_machine', params=params)
    assert r.status_code == 200, r.text
    return {i['crop_id'] for i in r.json()['items']}


@pytest.mark.parametrize('tab', [t for t in review_tabs.QUEUES if t != 'regions'])
def test_every_tab_serves_the_capture_range_as_date_specs(tab: str) -> None:
    assert {'taken_from', 'taken_to'} <= set(review_tabs.tab_filters(tab))
    served = {t['id']: t for t in review_tabs.review_tab_catalog()}[tab]
    kinds = {s['param']: s['kind'] for s in served['filter_specs']}
    assert kinds['taken_from'] == kinds['taken_to'] == 'date'


def test_the_date_specs_are_optional_with_no_default() -> None:
    for param in ('taken_from', 'taken_to'):
        assert FILTER_SPECS[param]['default'] is None
        assert FILTER_SPECS[param]['allows_unset'] is True


def test_the_range_narrows_the_queue(client: TestClient) -> None:
    assert _ids(client, taken_from='2024-03-01', taken_to='2024-03-31') == {'march'}
    assert _ids(client, taken_from='2024-04-01') == {'june'}
    assert _ids(client, taken_to='2024-03-05') == {'march'}


def test_a_bad_date_or_inverted_range_is_refused(client: TestClient) -> None:
    assert (
        client.get(f'{BASE}/check_machine', params={'taken_from': 'not-a-date'}).status_code == 422
    )
    inverted = client.get(
        f'{BASE}/check_machine', params={'taken_from': '2024-06-01', 'taken_to': '2024-03-01'}
    )
    assert inverted.status_code in (400, 422)
