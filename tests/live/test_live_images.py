"""Live: the image view model against a real OpenSearch.

The gallery's served counts must equal what the items index holds, the single-image view must list
exactly the items `GET /crops?image_id=` lists, and the pixel routes must serve the seeded files by
image id. The crop-anchored image routes are gone.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import INDEXES, refresh


pytestmark = pytest.mark.live

PAGE = 200


def _gallery(api_client: Any, **params: Any) -> list[dict[str, Any]]:
    """Every card of the gallery for ``params``, following the cursor."""
    cards: list[dict[str, Any]] = []
    cursor = None
    while True:
        query = {'limit': PAGE, **params, **({'cursor': cursor} if cursor else {})}
        resp = api_client.get('/images', params=query)
        assert resp.status_code == 200, resp.text
        page = resp.json()
        cards.extend(page['items'])
        cursor = page['next_cursor']
        if cursor is None:
            return cards


def _count(opensearch: Any, index: str, query: dict[str, Any]) -> int:
    resp = opensearch.post(f'/{index}/_count', json={'query': query})
    resp.raise_for_status()
    return int(resp.json()['count'])


def _live_items(include_test: bool) -> dict[str, Any]:
    must_not: list[dict[str, Any]] = [{'term': {'class_excluded': True}}]
    if not include_test:
        must_not.append({'term': {'test_holdout': True}})
    return {'bool': {'must_not': must_not}}


def test_gallery_counts_sum_to_the_items_index(api_client: Any, opensearch: Any) -> None:
    refresh(opensearch, INDEXES['items'])
    cards = _gallery(api_client)
    assert len({c['image_id'] for c in cards}) == len(cards), 'an image appears twice'
    assert len(cards) == _count(opensearch, INDEXES['images'], {'match_all': {}})
    assert sum(c['counts']['items'] for c in cards) == _count(
        opensearch, INDEXES['items'], _live_items(include_test=False)
    )
    with_test = _gallery(api_client, include_test=True)
    assert sum(c['counts']['items'] for c in with_test) == _count(
        opensearch, INDEXES['items'], _live_items(include_test=True)
    )
    for card in cards:
        counts = card['counts']
        assert counts['labeled'] + counts['unlabeled'] == counts['items'], card['image_id']
        assert card['thumbnail_url'].endswith(f'/images/{card["image_id"]}/thumbnail')


def test_gallery_sort_and_cursor_are_stable(api_client: Any) -> None:
    page = api_client.get('/images', params={'limit': 5}).json()
    assert page['sort_applied'] == 'indexed_at:desc'
    assert {o['id'] for o in page['sort_options']} >= {'indexed_at:asc', 'file_name:asc'}
    first = [c['image_id'] for c in _gallery(api_client, sort='file_name:asc')]
    again = [c['image_id'] for c in _gallery(api_client, sort='file_name:asc')]
    assert first == again
    assert len(set(first)) == len(first)
    assert api_client.get('/images', params={'sort': 'nope'}).status_code == 400


def test_image_detail_items_equal_the_crops_listing(api_client: Any) -> None:
    for card in _gallery(api_client, include_test=True)[:5]:
        image_id = card['image_id']
        detail = api_client.get(f'/images/{image_id}')
        assert detail.status_code == 200, detail.text
        body = detail.json()
        listed = api_client.get(
            '/crops',
            params={
                'image_id': image_id,
                'include_test': True,
                'include_excluded': True,
                'page_size': 500,
            },
        ).json()['crops']
        assert {i['crop_id'] for i in body['items']} == {c['crop_id'] for c in listed}, image_id
        assert {b['crop_id'] for b in body['boxes']} <= {i['crop_id'] for i in body['items']}
        assert body['image']['image_id'] == image_id
        assert body['image']['width']
        assert body['image']['height']


def test_pixels_are_served_by_image_id(api_client: Any) -> None:
    image_id = api_client.get('/images', params={'limit': 1}).json()['items'][0]['image_id']
    full = api_client.get(f'/images/{image_id}/file')
    assert full.status_code == 200
    assert full.headers['content-type'] == 'image/jpeg'
    scaled = api_client.get(f'/images/{image_id}/file', params={'max_dim': 128})
    assert scaled.status_code == 200
    assert len(scaled.content) <= len(full.content)
    thumb = api_client.get(f'/images/{image_id}/thumbnail', params={'size': 64})
    assert thumb.status_code == 200
    assert thumb.headers['content-type'] == 'image/jpeg'
    missing = api_client.get('/images/no-such-image/file')
    assert missing.status_code == 404
    assert missing.headers['content-type'].startswith('application/json')


def test_the_crop_anchored_image_routes_are_gone(api_client: Any, opensearch: Any) -> None:
    crop_id = api_client.get('/crops', params={'page_size': 1}).json()['crops'][0]['crop_id']
    assert api_client.get(f'/crops/{crop_id}/context').status_code == 404
    assert api_client.get(f'/crops/{crop_id}/image').status_code == 404
    assert api_client.get('/images/serve', params={'path': 'x.jpg'}).status_code == 404
