"""The image view: `GET /images` (gallery), `GET /images/{id}` (one image) and the pixel routes.

Counts equal an items-index aggregate, held-out and excluded items are hidden by default, an item
filter joins through the items, the sort options and the cursor are served, and the detail carries
every item with the boxes to draw (it replaces `GET /crops/{id}/context`).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.routers.curation import images_view as route
from src.schemas.curation.images import BoxView
from src.services.curation.wire import ITEM_WIRE_KEYS


if TYPE_CHECKING:
    from pathlib import Path

CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)
BASE = '/curation/projects/default'
ASC = {'sort': 'indexed_at:asc'}


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch, policy: str = 'store') -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    async def _policy() -> str:
        return policy

    monkeypatch.setattr(route, 'get_location_policy', _policy)
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _image(n: str, indexed: str, **extra: Any) -> dict[str, Any]:
    return {
        'image_id': n,
        'image_path': f'/data/{n}.jpg',
        'width': 640,
        'height': 480,
        'source': 'disk',
        'indexed_at': indexed,
        **extra,
    }


def _item(n: str, image: str, rank: int, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': n,
        'image_id': image,
        'crop_rank_in_image': rank,
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        **extra,
    }


def _world() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            ITEMS: {
                'a1': _item('a1', 'img-a', 1, class_id=3, class_name='sedan', class_validated=True),
                'a2': _item('a2', 'img-a', 2),
                'a3': _item('a3', 'img-a', 3, class_id=3, class_name='sedan', class_excluded=True),
                'b1': _item('b1', 'img-b', 1, class_id=4, class_name='truck'),
                'b2': _item('b2', 'img-b', 2, class_id=4, class_name='truck', test_holdout=True),
                'c1': _item('c1', 'img-c', 1, class_id=3, class_name='sedan'),
            },
            IMAGES: {
                'img-a': _image(
                    'img-a',
                    '2026-09-03T00:00:00',
                    near_dup_group_id='g1',
                    near_dup_group_size=2,
                    near_dup_is_representative=True,
                ),
                'img-b': _image('img-b', '2026-09-02T00:00:00'),
                'img-c': _image('img-c', '2026-09-01T00:00:00'),
                'img-d': _image('img-d', '2026-08-01T00:00:00'),
            },
        }
    )


def _cards(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c['image_id']: c for c in body['items']}


def test_gallery_counts_equal_an_items_aggregate_and_hide_holdout_and_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _world()
    # The fake sorts ascending whatever `order` says; the default's direction is pinned below.
    body = _client(fake, monkeypatch).get(f'{BASE}/images', params=ASC).json()
    cards = _cards(body)
    assert [c['image_id'] for c in body['items']] == ['img-d', 'img-c', 'img-b', 'img-a']
    assert cards['img-a']['counts'] == {
        'items': 2,
        'labeled': 1,
        'validated': 1,
        'unlabeled': 1,
        'in_holdout': 0,
    }
    # img-b: the held-out item is hidden from `items` but still counted as held out.
    assert cards['img-b']['counts']['items'] == 1
    assert cards['img-b']['counts']['in_holdout'] == 1
    assert cards['img-d']['counts']['items'] == 0
    visible = [
        d
        for d in fake.docs(ITEMS).values()
        if not d.get('test_holdout') and not d.get('class_excluded')
    ]
    assert sum(c['counts']['items'] for c in body['items']) == len(visible)
    assert body['hidden'] == {'test_holdout': 1, 'excluded': 1}
    assert body['total'] == 4
    assert body['join_truncated'] is False
    assert body['empty_reason'] is None


def test_hidden_counts_follow_the_item_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)
    sedans = client.get(f'{BASE}/images', params={'class_name': 'sedan', **ASC}).json()
    trucks = client.get(f'{BASE}/images', params={'class_name': 'truck', **ASC}).json()
    assert sedans['hidden'] == {'test_holdout': 0, 'excluded': 1}
    assert trucks['hidden'] == {'test_holdout': 1, 'excluded': 0}


def test_include_test_counts_and_draws_the_held_out_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _client(_world(), monkeypatch).get(
        f'{BASE}/images', params={'include_test': 'true', **ASC}
    )
    card = _cards(body.json())['img-b']
    assert card['counts']['items'] == 2
    assert {b['crop_id']: b['tone'] for b in card['boxes']} == {'b1': 'machine', 'b2': 'holdout'}
    assert body.json()['hidden']['test_holdout'] == 0


def test_a_page_is_three_queries_whatever_its_size(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _world()
    _client(fake, monkeypatch).get(f'{BASE}/images', params={'limit': 4})
    # the images, their item facts, and the hidden counts over the matching items
    assert fake.searched_indexes == [IMAGES, ITEMS, ITEMS]


def test_card_boxes_carry_label_tone_and_image_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _world()
    fake.docs(ITEMS)['a1']['region_boxes'] = [
        {'box_id': 'r1', 'bbox_norm': [0.2, 0.2, 0.3, 0.3], 'state': 'accepted'},
        {'box_id': 'r2', 'bbox_norm': [0.6, 0.6, 0.7, 0.7], 'state': 'rejected'},
    ]
    card = _cards(_client(fake, monkeypatch).get(f'{BASE}/images').json())['img-a']
    boxes = [BoxView.model_validate(b) for b in card['boxes']]
    by_key = {(b.crop_id, b.box_id): b for b in boxes}
    assert by_key[('a1', None)].label == 'sedan'
    assert by_key[('a1', None)].tone == 'validated'
    assert by_key[('a1', None)].kind == 'item'
    assert by_key[('a1', 'r1')].kind == 'region'
    assert by_key[('a1', 'r1')].bbox_norm == (0.2, 0.2, 0.3, 0.3)
    assert by_key[('a1', 'r2')].tone == 'rejected'
    assert by_key[('a2', None)].label == 'unlabeled'
    assert by_key[('a2', None)].tone == 'unlabeled'
    assert by_key[('a2', None)].class_name is None
    assert ('a3', None) not in by_key, 'an excluded item is not drawn'
    assert card['boxes_truncated'] is False
    assert card['near_dup'] == {
        'group_id': 'g1',
        'purpose': 'dedup',
        'size': 2,
        'is_representative': True,
    }
    assert card['file_name'] == 'img-a.jpg'
    assert card['thumbnail_url'] == f'{BASE}/images/img-a/thumbnail'


def test_an_item_filter_joins_through_the_items(monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        _client(_world(), monkeypatch).get(f'{BASE}/images', params={'class_name': 'truck'}).json()
    )
    assert [c['image_id'] for c in body['items']] == ['img-b']
    assert body['total'] == 1


def test_join_truncated_says_when_the_join_used_only_part_of_the_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _world()
    real = fake.search

    async def _search(**kw: Any) -> Any:
        resp = await real(**kw)
        agg = (resp.get('aggregations') or {}).get('images')
        if kw['index'] == ITEMS and agg is not None and 'aggs' not in kw['body']['aggs']['images']:
            agg['sum_other_doc_count'] = 7
        return resp

    fake.search = _search
    body = _client(fake, monkeypatch).get(f'{BASE}/images', params={'class_name': 'sedan'}).json()
    assert body['join_truncated'] is True


def test_label_state_joins(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)

    def ids(state: str) -> list[str]:
        r = client.get(f'{BASE}/images', params={'label_state': state, **ASC})
        assert r.status_code == 200, r.text
        return [c['image_id'] for c in r.json()['items']]

    assert ids('has_unlabeled') == ['img-a']
    assert ids('all_labeled') == ['img-c', 'img-b']
    assert ids('no_items') == ['img-d']
    bad = client.get(f'{BASE}/images', params={'label_state': 'no_items', 'class_name': 'x'})
    assert bad.status_code == 400


def test_in_holdout_and_near_dup_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)

    def ids(**params: str) -> list[str]:
        got = client.get(f'{BASE}/images', params={**params, **ASC}).json()
        return [c['image_id'] for c in got['items']]

    assert ids(in_holdout='true') == ['img-b']
    assert ids(in_holdout='false') == ['img-d', 'img-c', 'img-a']
    assert ids(near_dup='representative') == ['img-a']


def test_sort_options_are_served_and_an_unknown_sort_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(_world(), monkeypatch)
    body = client.get(f'{BASE}/images').json()
    assert body['sort_applied'] == 'indexed_at:desc'
    assert [o['id'] for o in body['sort_options']] == [
        'indexed_at:desc',
        'indexed_at:asc',
        'taken_at:desc',
        'taken_at:asc',
        'file_name:asc',
    ]
    assert all(o['label'] for o in body['sort_options'])
    asc = client.get(f'{BASE}/images', params=ASC).json()
    assert [c['image_id'] for c in asc['items']] == ['img-d', 'img-c', 'img-b', 'img-a']
    assert client.get(f'{BASE}/images', params={'sort': 'bogus'}).status_code == 400


def test_the_cursor_pages_without_repeats_or_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(5):
        params: dict[str, Any] = {'limit': 3, **ASC}
        if cursor:
            params['cursor'] = cursor
        page = client.get(f'{BASE}/images', params=params).json()
        seen += [c['image_id'] for c in page['items']]
        # (the fake counts after search_after; OpenSearch's total ignores it)
        cursor = page['next_cursor']
        if cursor is None:
            break
    assert seen == ['img-d', 'img-c', 'img-b', 'img-a']
    other_sort = client.get(
        f'{BASE}/images', params={'limit': 1, 'sort': 'indexed_at:asc', 'cursor': 'bm9wZQ=='}
    )
    assert other_sort.status_code == 400
    first = client.get(f'{BASE}/images', params={'limit': 1}).json()['next_cursor']
    mismatched = client.get(f'{BASE}/images', params={'cursor': first, 'sort': 'indexed_at:asc'})
    assert mismatched.status_code == 400
    assert mismatched.json()['detail']['error'] == 'cursor_invalid'


def test_empty_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: {}, IMAGES: {}}), monkeypatch)
    assert client.get(f'{BASE}/images').json()['empty_reason'] == 'no_images'
    assert client.get(f'{BASE}/images', params={'source': 'x'}).json()['empty_reason'] == 'no_match'


def test_location_filters_follow_the_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _world()
    kept = _client(fake, monkeypatch).get(
        f'{BASE}/images', params={'has_location': 'true', 'geo_bbox': '7,46,9,48'}
    )
    assert kept.status_code == 200, kept.text
    dropped = _client(fake, monkeypatch, policy='drop').get(
        f'{BASE}/images', params={'has_location': 'true'}
    )
    assert dropped.json()['items'] == []
    bad = _client(fake, monkeypatch).get(f'{BASE}/images', params={'geo_bbox': 'nope'})
    assert bad.status_code == 422
    assert 'geo_bbox needs west,south,east,north in degrees' in bad.text


# --- GET /images/{image_id} -------------------------------------------------------------------


def _detail_fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            ITEMS: {
                'a1': _item(
                    'a1',
                    'img-a',
                    1,
                    class_id=3,
                    class_name='sedan',
                    class_validated=True,
                    region_status='verify_rejected',
                    region_boxes=[
                        {'box_id': 'b1', 'bbox_norm': [0.15, 0.15, 0.2, 0.2], 'state': 'accepted'},
                        {
                            'box_id': 'b2',
                            'bbox_norm': [0.5, 0.5, 0.6, 0.6],
                            'state': 'rejected',
                            'rejection_reason': 'verifier_no_verdict',
                        },
                    ],
                ),
                'a2': _item('a2', 'img-a', 2, test_holdout=True),
                'b1': _item('b1', 'img-b', 1),
            },
            IMAGES: {
                'img-a': {
                    **_image('img-a', '2026-09-01T00:00:00', dataset_split='val'),
                    'width': 1920,
                    'height': 1080,
                    'capture': {
                        'taken_local': '2026-08-01T10:00:00',
                        'has_location': True,
                        'location': {'lat': 47.123456, 'lon': 8.654321},
                    },
                    'pe_embedding': [0.0] * 4,
                },
                'img-b': _image('img-b', '2026-09-02T00:00:00'),
            },
        }
    )


def test_image_detail_serves_the_image_its_items_and_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = _client(_detail_fake(), monkeypatch).get(f'{BASE}/images/img-a')
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {'image', 'counts', 'items', 'boxes', 'items_truncated'}
    image = body['image']
    assert (image['width'], image['height'], image['file_name']) == (1920, 1080, 'img-a.jpg')
    assert image['file_url'] == f'{BASE}/images/img-a/file'
    assert image['taken_at'] == '2026-08-01T10:00:00'
    assert image['dataset_split'] == 'val'
    assert image['in_holdout'] is True
    assert image['location'] == {'latitude': 47.123456, 'longitude': 8.654321, 'precision': None}
    assert [i['crop_id'] for i in body['items']] == ['a1', 'a2']
    assert all(set(i) == ITEM_WIRE_KEYS for i in body['items'])
    assert body['counts'] == {
        'items': 2,
        'labeled': 1,
        'validated': 1,
        'unlabeled': 1,
        'in_holdout': 1,
    }
    boxes = {(b['crop_id'], b['box_id']): b for b in body['boxes']}
    assert boxes[('a1', None)]['bbox_norm'] == [0.1, 0.1, 0.5, 0.5]
    assert boxes[('a1', 'b1')]['bbox_norm'] == [0.15, 0.15, 0.2, 0.2]
    assert boxes[('a1', 'b2')]['tone'] == 'rejected'
    assert boxes[('a2', None)]['tone'] == 'holdout'
    assert body['items_truncated'] is False


def test_image_detail_location_follows_the_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _detail_fake()
    dropped = _client(fake, monkeypatch, policy='drop').get(f'{BASE}/images/img-a').json()
    assert dropped['image']['location'] is None
    coarse = _client(fake, monkeypatch, policy='store_coarse').get(f'{BASE}/images/img-a').json()
    assert coarse['image']['location'] == {
        'latitude': 47.12,
        'longitude': 8.65,
        'precision': 'coarse',
    }


def test_image_detail_unknown_image_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _client(_detail_fake(), monkeypatch).get(f'{BASE}/images/nope').status_code == 404


def test_image_detail_fills_missing_pixel_size_from_the_file_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.config.curation as curation_config_mod
    import src.config.curation_accessors as curation_accessors_mod

    img_path = tmp_path / 'a.jpg'
    Image.new('RGB', (321, 233)).save(img_path, format='JPEG')
    monkeypatch.setattr(
        curation_accessors_mod,
        '_default_curation_config',
        curation_config_mod.CurationConfig(source_root=tmp_path),
    )
    fake = QueryFakeOpenSearch(
        {
            ITEMS: {},
            IMAGES: {
                'img-a': {
                    **_image('img-a', '2026-09-01T00:00:00'),
                    'image_path': str(img_path),
                    'width': None,
                    'height': None,
                }
            },
        }
    )
    image = _client(fake, monkeypatch).get(f'{BASE}/images/img-a').json()['image']
    assert (image['width'], image['height']) == (321, 233)


@pytest.mark.parametrize('exc', [SyntaxError('broken PNG chunk'), TypeError('malformed EXIF')])
def test_pixel_size_from_header_is_none_for_every_pillow_decode_error(
    exc: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.config.curation as curation_config_mod
    import src.config.curation_accessors as curation_accessors_mod
    from src.services.curation.image_detail import _pixel_size_from_header

    img_path = tmp_path / 'a.png'
    Image.new('RGB', (4, 4)).save(img_path, format='PNG')
    monkeypatch.setattr(
        curation_accessors_mod,
        '_default_curation_config',
        curation_config_mod.CurationConfig(source_root=tmp_path),
    )

    def _broken(*_a: Any, **_k: Any) -> Any:
        raise exc

    monkeypatch.setattr(Image, 'open', _broken)
    assert _pixel_size_from_header(str(img_path)) is None


def test_detail_items_carry_their_outline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake does not apply ``_source.excludes``, so this pins what the route asks for."""
    fake = _detail_fake()
    asked: dict[str, Any] = {}
    real = fake.search

    async def _record(**kwargs: Any) -> Any:
        if kwargs['index'] == ITEMS and 'excludes' in kwargs['body'].get('_source', {}):
            asked.update(kwargs['body']['_source'])
        return await real(**kwargs)

    fake.search = _record
    _client(fake, monkeypatch).get(f'{BASE}/images/img-a')
    assert 'mask_polygon' not in asked['excludes']
    assert 'class_id_history' in asked['excludes']


def test_the_documented_detail_model_matches_what_is_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation._image_view_models import ImageDetail

    client = _client(_detail_fake(), monkeypatch)
    served = client.get(f'{BASE}/images/img-a').json()
    assert set(served) == set(ImageDetail.model_fields)
    ImageDetail.model_validate(served)
    ref = client.app.openapi()['paths']['/curation/projects/{project}/images/{image_id}']['get'][  # type: ignore[attr-defined]
        'responses'
    ]['200']['content']['application/json']['schema']
    assert ref['$ref'].endswith('/ImageDetail')


# --- the pixels -------------------------------------------------------------------------------


def _pixels_client(
    monkeypatch: pytest.MonkeyPatch, root: Path, docs: dict[str, dict[str, Any]]
) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import images as images_mod
    from src.services.curation import image_serving

    cfg = dataclasses.replace(base_curation_config(), source_root=root)
    monkeypatch.setattr(image_serving, 'get_curation_config', lambda: cfg)
    monkeypatch.setattr(images_mod, 'THUMBNAIL_CACHE', image_serving.ThumbnailCache())
    app = FastAPI()
    mount_curation_routers(app, images_mod.router)
    fake = QueryFakeOpenSearch({IMAGES: docs})
    app.dependency_overrides[images_mod._raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_file_and_thumbnail_serve_clean_pixels_by_image_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Image.new('RGB', (400, 200), (200, 10, 10)).save(tmp_path / 'a.jpg')
    client = _pixels_client(
        monkeypatch,
        tmp_path,
        {'img-a': {'image_id': 'img-a', 'image_path': str(tmp_path / 'a.jpg')}},
    )
    full = client.get(f'{BASE}/images/img-a/file')
    assert full.status_code == 200
    assert full.headers['content-type'] == 'image/jpeg'
    import io

    assert Image.open(io.BytesIO(full.content)).size == (400, 200)
    scaled = client.get(f'{BASE}/images/img-a/file', params={'max_dim': 128})
    assert max(Image.open(io.BytesIO(scaled.content)).size) == 128
    thumb = client.get(f'{BASE}/images/img-a/thumbnail', params={'size': 64})
    assert thumb.status_code == 200
    # The whole frame, aspect kept: boxes in image coordinates land on the same place.
    assert Image.open(io.BytesIO(thumb.content)).size == (64, 32)


def test_pixels_of_an_image_the_project_does_not_hold_are_a_404_with_no_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Image.new('RGB', (4, 4)).save(tmp_path / 'theirs.jpg')
    client = _pixels_client(monkeypatch, tmp_path, {})
    for suffix in ('file', 'thumbnail'):
        r = client.get(f'{BASE}/images/theirs/{suffix}')
        assert r.status_code == 404
        assert r.headers['content-type'].startswith('application/json')


def test_pixels_of_a_vanished_file_answer_the_typed_source_missing_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _pixels_client(
        monkeypatch, tmp_path, {'gone': {'image_id': 'gone', 'image_path': str(tmp_path / 'x.jpg')}}
    )
    for suffix in ('file', 'thumbnail'):
        r = client.get(f'{BASE}/images/gone/{suffix}')
        assert r.status_code == 404, r.text
        detail = r.json()['detail']
        assert detail['error'] == 'source_missing'
        assert detail['root_mounted'] is True


def test_the_default_sort_is_newest_indexed_first_with_the_image_id_as_tiebreak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _world()
    bodies: list[dict[str, Any]] = []
    real = fake.search

    async def _record(**kw: Any) -> Any:
        if kw['index'] == IMAGES:
            bodies.append(kw['body'])
        return await real(**kw)

    fake.search = _record
    _client(fake, monkeypatch).get(f'{BASE}/images')
    assert bodies[0]['sort'] == [{'indexed_at': {'order': 'desc'}}, {'image_id': {'order': 'asc'}}]
