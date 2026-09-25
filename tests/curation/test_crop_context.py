"""Image context for an item, and the review-dismissal list.

- ``GET /crops/{id}/context``: the item's source-image metadata and every
  item from the same image (shared wire format), for a context panel.
- ``GET /crops?review_dismissed=true`` lists items hidden from review;
  ``POST /crops/{id}/review_undismiss`` returns one to the queues;
  items carry ``review_dismissed_at``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch


if TYPE_CHECKING:
    from pathlib import Path

    import pytest
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


def test_crop_image_context_carries_full_drawing_geometry_for_every_item() -> None:
    """/context must serve every box a full-image
    labeling view needs, in source-image-normalized coordinates, plus
    class + status + reason + validation fields, for every sibling item.
    """
    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'a1': {
                    'crop_id': 'a1',
                    'image_id': 'img-a',
                    'crop_rank_in_image': 1,
                    'bbox_norm': [0.1, 0.1, 0.4, 0.4],
                    'region_bbox_norm': [0.15, 0.15, 0.2, 0.2],
                    'region_candidate_bbox_norm': [0.5, 0.5, 0.6, 0.6],
                    'class_id': 3,
                    'class_name': 'sedan',
                    'class_validated': True,
                    'region_status': 'verify_rejected',
                    'region_rejection_reason': 'verifier_no_verdict',
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
                }
            },
        }
    )
    body = _client(fake).get('/curation/crops/a1/context').json()
    item = body['items'][0]
    assert item['bbox_norm'] == [0.1, 0.1, 0.4, 0.4]
    assert item['region_bbox_norm'] == [0.15, 0.15, 0.2, 0.2]
    assert item['region_candidate_bbox_norm'] == [0.5, 0.5, 0.6, 0.6]
    assert item['class_id'] == 3
    assert item['class_name'] == 'sedan'
    assert item['class_validated'] is True
    assert item['region_status'] == 'verify_rejected'
    assert item['region_rejection_reason'] == 'verifier_no_verdict'
    assert body['image']['width'] == 1920
    assert body['image']['height'] == 1080


def test_crop_image_context_fills_missing_pixel_size_from_the_file_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Width/height must never be left null when the
    image is actually servable -- fall back to reading the file header."""
    import src.config.curation as curation_config_mod

    img_path = tmp_path / 'a.jpg'
    Image.new('RGB', (321, 233), color=(10, 20, 30)).save(img_path, format='JPEG')
    patched_cfg = curation_config_mod.CurationConfig(source_root=tmp_path)
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', patched_cfg)

    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'a1': {'crop_id': 'a1', 'image_id': 'img-a', 'crop_rank_in_image': 1},
            },
            IMAGES: {
                'img-a': {
                    'image_id': 'img-a',
                    'image_path': str(img_path),
                    'width': None,
                    'height': None,
                    'source': 'disk_a',
                    'indexed_at': '2026-09-01T00:00:00+00:00',
                }
            },
        }
    )
    body = _client(fake).get('/curation/crops/a1/context').json()
    assert body['image']['width'] == 321
    assert body['image']['height'] == 233


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
