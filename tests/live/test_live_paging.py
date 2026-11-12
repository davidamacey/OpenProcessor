"""Live: `GET /crops` pages by cursor past the 10,000-row window, and `/crops/locate` agrees with it.

The offline suite proves the cursor arithmetic against a double; only a real OpenSearch refuses
`from + size` past `index.max_result_window`, so this is the check that the window is really gone.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from .conftest import INDEXES, get_doc, refresh


pytestmark = pytest.mark.live

CLONES = 10_100
PREFIX = 'paging_clone_'
BATCH = 2_000
LIMIT = 500


@pytest.fixture(scope='module')
def cohort(opensearch: Any) -> Iterator[list[str]]:
    """``CLONES`` copies of one seeded crop, each with its own ``confidence`` so the sort has ties."""
    source = get_doc(opensearch, INDEXES['items'], 'cls0_0000')['_source']
    ids = [f'{PREFIX}{i:06d}' for i in range(CLONES)]
    for start in range(0, CLONES, BATCH):
        lines: list[str] = []
        for i, doc_id in enumerate(ids[start : start + BATCH], start=start):
            lines += [
                json.dumps({'index': {'_index': INDEXES['items'], '_id': doc_id}}),
                json.dumps({**source, 'crop_id': doc_id, 'confidence': round((i % 50) / 50, 2)}),
            ]
        resp = opensearch.post(
            '/_bulk',
            content='\n'.join(lines) + '\n',
            headers={'Content-Type': 'application/x-ndjson'},
            timeout=120.0,
        )
        resp.raise_for_status()
        assert resp.json()['errors'] is False, resp.text[:500]
    refresh(opensearch, INDEXES['items'])
    try:
        yield ids
    finally:
        opensearch.post(
            f'/{INDEXES["items"]}/_delete_by_query',
            params={'refresh': 'true'},
            json={'query': {'prefix': {'crop_id': PREFIX}}},
            timeout=120.0,
        ).raise_for_status()


def _read_all(api_client: Any, **params: Any) -> tuple[list[str], int]:
    seen: list[str] = []
    cursor = None
    total = 0
    while True:
        query = {'limit': LIMIT, **params, **({'cursor': cursor} if cursor else {})}
        resp = api_client.get('/crops', params=query)
        assert resp.status_code == 200, resp.text
        page = resp.json()
        total = page['total']
        seen += [c['id'] for c in page['crops']]
        cursor = page['next_cursor']
        if cursor is None:
            return seen, total


def test_crops_cursor_reads_past_the_window(api_client: Any, cohort: list[str]) -> None:
    seen, total = _read_all(api_client, sort='confidence:desc')
    assert total >= CLONES
    assert len(seen) == total, 'the cursor walk must reach every row'
    assert len(set(seen)) == total, 'no row twice'
    assert total > 10_000

    for position in (0, 250, 10_050, total - 1):
        target = seen[position]
        loc = api_client.get('/crops/locate', params={'crop_id': target, 'sort': 'confidence:desc'})
        assert loc.status_code == 200, loc.text
        body = loc.json()
        assert (body['in_view'], body['index'], body['total']) == (True, position, total)
        if position:
            page = api_client.get(
                '/crops',
                params={'sort': 'confidence:desc', 'limit': 5, 'cursor': body['cursor_before']},
            ).json()
            assert [c['id'] for c in page['crops']] == seen[position : position + 5]
        after = api_client.get(
            '/crops',
            params={'sort': 'confidence:desc', 'limit': 5, 'cursor': body['cursor_after']},
        ).json()
        assert [c['id'] for c in after['crops']] == seen[position + 1 : position + 6]


def test_locate_reports_a_crop_the_filter_hides(api_client: Any, cohort: list[str]) -> None:
    body = api_client.get(
        '/crops/locate', params={'crop_id': cohort[0], 'class_name': 'no-such-class'}
    ).json()
    assert (body['in_view'], body['index'], body['reason']) == (False, None, 'filtered_out')
