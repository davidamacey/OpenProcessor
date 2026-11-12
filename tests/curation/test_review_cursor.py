"""`GET /review/{tab}` pages by cursor and reads past the 10,000-row window."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.paging_fakes import PagingOpenSearch, crops_client
from src.config.curation_accessors import base_curation_config


if TYPE_CHECKING:
    from fastapi.testclient import TestClient

ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default'
N = 10_050


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    docs = {
        f'c{i:05d}': {
            'crop_id': f'c{i:05d}',
            'image_id': f'img-{i}',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_source': 'vlm_unmatched',
            'updated_at': f'2026-01-01T00:{i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}',
        }
        for i in range(N)
    }
    return crops_client(PagingOpenSearch({ITEMS: docs}), monkeypatch)


def test_queue_reads_past_ten_thousand(client: TestClient) -> None:
    seen: list[str] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {'limit': 200, 'sort': 'recent'}
        if cursor:
            params['cursor'] = cursor
        r = client.get(f'{BASE}/review/to_label', params=params)
        assert r.status_code == 200, r.text
        page = r.json()
        assert 'page' not in page
        assert page['limit'] == 200
        assert page['total'] == N
        seen += [i['id'] for i in page['items']]
        cursor = page['next_cursor']
        if cursor is None:
            break
    assert len(seen) == N
    assert len(set(seen)) == N


def test_cursor_of_another_sort_is_refused(client: TestClient) -> None:
    first = client.get(f'{BASE}/review/to_label', params={'limit': 5, 'sort': 'recent'}).json()
    r = client.get(
        f'{BASE}/review/to_label',
        params={'cursor': first['next_cursor'], 'sort': 'atypicality'},
    )
    assert r.status_code == 400
    assert r.json()['detail']['error'] == 'cursor_invalid'
