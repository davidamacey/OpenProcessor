"""Totals and hidden counts are paid once, on the first page, then ride in the cursor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.paging_fakes import AggregatingPagingOpenSearch, images_client
from src.config.curation_accessors import base_curation_config


if TYPE_CHECKING:
    from fastapi.testclient import TestClient

ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default'
N = 23


def _world() -> dict[str, dict[str, dict[str, Any]]]:
    items = {}
    for i in range(N):
        items[f'c{i:03d}'] = {
            'crop_id': f'c{i:03d}',
            'image_id': f'img{i:03d}',
            'image_path': f'/data/{i:03d}.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'updated_at': f'2026-02-{(i % 9) + 1:02d}T00:00:00Z',
            'test_holdout': i % 10 == 0,
        }
    return {ITEMS: items}


@pytest.fixture
def fake() -> AggregatingPagingOpenSearch:
    return AggregatingPagingOpenSearch(_world())


@pytest.fixture
def client(fake: AggregatingPagingOpenSearch, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return images_client(fake, monkeypatch)


def _walk(client: TestClient, fake: Any, url: str, rows_key: str, **params: Any) -> list[tuple]:
    """``(body, searches issued)`` per page, following ``next_cursor`` to the end."""
    pages, cursor = [], None
    while True:
        before = len(fake.bodies)
        q = {**params, **({'cursor': cursor} if cursor else {})}
        r = client.get(url, params=q)
        assert r.status_code == 200, r.text
        body = r.json()
        pages.append((body, fake.bodies[before:]))
        cursor = body['next_cursor']
        if not cursor:
            return pages


def test_crops_cursor_pages_search_once_without_totals(
    client: TestClient, fake: AggregatingPagingOpenSearch
) -> None:
    pages = _walk(client, fake, f'{BASE}/crops', 'crops', limit=5)
    assert len(pages) > 2
    first, first_searches = pages[0]
    assert any(b.get('track_total_hits') is True for b in first_searches)
    for body, searches in pages[1:]:
        assert len(searches) == 1
        assert searches[0]['track_total_hits'] is False
        assert 'aggs' not in searches[0]
        assert body['total'] == first['total']
        assert body['hidden'] == first['hidden']
    assert first['hidden']['test_holdout'] > 0


def test_table_cursor_pages_search_once_without_totals(
    client: TestClient, fake: AggregatingPagingOpenSearch
) -> None:
    pages = _walk(client, fake, f'{BASE}/table', 'items', rows='detections', limit=5)
    assert len(pages) > 2
    first = pages[0][0]
    for body, searches in pages[1:]:
        assert len(searches) == 1
        assert searches[0]['track_total_hits'] is False
        assert 'aggs' not in searches[0]
        assert body['total'] == first['total']
        assert body['hidden'] == first['hidden']


def test_a_cursor_without_counts_counts_for_itself(
    client: TestClient, fake: AggregatingPagingOpenSearch
) -> None:
    from src.services.curation.list_cursor import encode_cursor

    cursor = encode_cursor('updated_at:desc', ['2026-02-05T00:00:00Z', 'c004'])
    r = client.get(f'{BASE}/crops', params={'cursor': cursor, 'limit': 5})
    assert r.status_code == 200, r.text
    assert r.json()['total'] > 0
