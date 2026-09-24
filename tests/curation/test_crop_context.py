"""Image context for an item, and the review-dismissal list.

- ``GET /crops/{id}/context``: the item's source-image metadata and every
  item from the same image (shared wire format), for a context panel.
- ``GET /crops?review_dismissed=true`` lists items hidden from review;
  ``POST /crops/{id}/review_undismiss`` returns one to the queues;
  items carry ``review_dismissed_at``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, get_curation_config, index_name
from src.services.curation.wire import ITEM_WIRE_KEYS


CFG = get_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)


def _client(fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            ITEMS: {
                'a1': {'crop_id': 'a1', 'image_id': 'img-a', 'crop_rank_in_image': 1},
                'a2': {'crop_id': 'a2', 'image_id': 'img-a', 'crop_rank_in_image': 2},
                'b1': {
                    'crop_id': 'b1',
                    'image_id': 'img-b',
                    'review_dismissed_at': '2026-09-01T00:00:00+00:00',
                    'review_dismissed_by': 'human',
                },
            },
            IMAGES: {
                'img-a': {
                    'image_id': 'img-a',
                    'image_path': '/data/a.jpg',
                    'width': 1920,
                    'height': 1080,
                    'source': 'disk_a',
                    'indexed_at': '2026-09-01T00:00:00+00:00',
                    'pe_embedding': [0.0] * 4,
                }
            },
        }
    )


def test_crop_image_context() -> None:
    r = _client(_fake()).get('/curation/crops/a1/context')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['image'] == {
        'image_id': 'img-a',
        'image_path': '/data/a.jpg',
        'width': 1920,
        'height': 1080,
        'source': 'disk_a',
        'indexed_at': '2026-09-01T00:00:00+00:00',
    }
    assert [i['crop_id'] for i in body['items']] == ['a1', 'a2']
    assert all(set(i) == ITEM_WIRE_KEYS for i in body['items'])


def test_crop_image_context_unknown_crop_is_404() -> None:
    assert _client(_fake()).get('/curation/crops/nope/context').status_code == 404


def test_dismissed_list_and_undismiss() -> None:
    fake = _fake()
    client = _client(fake)
    r = client.get('/curation/crops', params={'review_dismissed': 'true'})
    assert [c['crop_id'] for c in r.json()['crops']] == ['b1']
    assert r.json()['crops'][0]['review_dismissed_at'] == '2026-09-01T00:00:00+00:00'

    r = client.post('/curation/crops/b1/review_undismiss')
    assert r.status_code == 200, r.text
    assert r.json()['review_dismissed_at'] is None
    assert fake.docs(ITEMS)['b1'].get('review_dismissed_at') is None
    r = client.get('/curation/crops', params={'review_dismissed': 'true'})
    assert r.json()['crops'] == []
