"""Live: the gallery's stored per-image counts follow label writes (R-13).

The backfill counts every image; a label write marks its image, the 5-second beat drain recounts
it, and ``GET /images?sort=unlabeled_count:desc`` then orders by the new counts. Waits are bounded
by ``wait_until`` deadlines, never by a fixed sleep.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import CLASS_NAMES, INDEXES, refresh, search, wait_job, wait_until


pytestmark = pytest.mark.live

SORT = 'unlabeled_count:desc'
DRAIN_DEADLINE_S = 60.0
LABELLED = 3


def _all_cards(api_client: Any, **params: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every card for ``params`` in served order, and the first page's body."""
    cards: list[dict[str, Any]] = []
    cursor = None
    first: dict[str, Any] = {}
    while True:
        query = {'limit': 200, **params, **({'cursor': cursor} if cursor else {})}
        resp = api_client.get('/images', params=query)
        assert resp.status_code == 200, resp.text
        page = resp.json()
        first = first or page
        cards.extend(page['items'])
        cursor = page['next_cursor']
        if cursor is None:
            return cards, first


def _counts_served(api_client: Any) -> bool:
    page = api_client.get('/images', params={'limit': 1}).json()
    return SORT in {o['id'] for o in page['sort_options']}


def test_the_sort_is_served_after_the_backfill(api_client: Any) -> None:
    started = api_client.post('/image_counts/backfill')
    assert started.status_code in (202, 409), started.text
    if started.status_code == 202:
        job = wait_job(api_client, started.json()['job_id'], timeout=180.0)
        assert job['status'] == 'completed', job
    else:
        assert started.json()['detail']['error'] == 'job_busy'
    assert wait_until(lambda: _counts_served(api_client), timeout=DRAIN_DEADLINE_S)


def test_labelling_three_items_reorders_the_unlabeled_sort_after_the_drain(
    api_client: Any, opensearch: Any, class_ids: dict[str, int]
) -> None:
    assert wait_until(lambda: _counts_served(api_client), timeout=DRAIN_DEADLINE_S), (
        'the unlabeled_count sort was never served: run the backfill test first'
    )
    cards, _first = _all_cards(api_client, sort=SORT)
    target = cards[0]
    assert target['counts']['unlabeled'] >= LABELLED, 'the seed has no image with 3 unlabeled items'

    refresh(opensearch, INDEXES['items'])
    hits = search(
        opensearch,
        INDEXES['items'],
        {
            'size': LABELLED,
            '_source': False,
            'sort': [{'crop_id': 'asc'}],
            'query': {
                'bool': {
                    'filter': [{'term': {'image_id': target['image_id']}}],
                    'must_not': [
                        {'exists': {'field': 'class_id'}},
                        {'term': {'class_excluded': True}},
                        {'term': {'test_holdout': True}},
                    ],
                }
            },
        },
    )['hits']['hits']
    assert len(hits) == LABELLED
    for hit in hits:
        resp = api_client.put(
            f'/crops/{hit["_id"]}/label',
            json={'class_id': class_ids[CLASS_NAMES[3]], 'label_source': 'human'},
        )
        assert resp.status_code == 200, resp.text

    def drained() -> list[dict[str, Any]] | None:
        served, _ = _all_cards(api_client, sort=SORT)
        by_id = {c['image_id']: c for c in served}
        stored_moved = by_id[target['image_id']]['counts']['unlabeled'] == (
            target['counts']['unlabeled'] - LABELLED
        )
        counts = [c['counts']['unlabeled'] for c in served]
        ordered = counts == sorted(counts, reverse=True)
        return served if stored_moved and ordered else None

    served = wait_until(drained, timeout=DRAIN_DEADLINE_S)
    assert served, 'the served order never caught up with the labels within the drain deadline'
    assert served[0]['counts']['unlabeled'] == max(c['counts']['unlabeled'] for c in served)
