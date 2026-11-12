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
MAX_LABELLED = 20


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


def _swap_pair(cards: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Adjacent cards (above, below) in the served order and how many items of ``above`` to
    label so its count drops strictly under ``below``'s: the cheapest such pair."""
    best: tuple[int, int] | None = None
    for i in range(len(cards) - 1):
        above, below = cards[i]['counts']['unlabeled'], cards[i + 1]['counts']['unlabeled']
        need = above - below + 1
        if below >= 1 and need <= MAX_LABELLED and (best is None or need < best[1]):
            best = (i, need)
    assert best is not None, 'the seed has no two adjacent images that a few labels can swap'
    return cards[best[0]], cards[best[0] + 1], best[1]


def _position(cards: list[dict[str, Any]], image_id: str) -> int:
    return [c['image_id'] for c in cards].index(image_id)


# After the int merge, send the label writes through request_when_free (tests/live/conftest.py).
def test_labelling_items_swaps_two_images_in_the_unlabeled_sort_after_the_drain(
    api_client: Any, opensearch: Any, class_ids: dict[str, int]
) -> None:
    assert wait_until(lambda: _counts_served(api_client), timeout=DRAIN_DEADLINE_S), (
        'the unlabeled_count sort was never served: run the backfill test first'
    )
    before, _first = _all_cards(api_client, sort=SORT)
    above, below, labelled = _swap_pair(before)
    start_above = _position(before, above['image_id'])
    start_below = _position(before, below['image_id'])
    assert start_below == start_above + 1

    refresh(opensearch, INDEXES['items'])
    hits = search(
        opensearch,
        INDEXES['items'],
        {
            'size': labelled,
            '_source': False,
            'sort': [{'crop_id': 'asc'}],
            'query': {
                'bool': {
                    'filter': [{'term': {'image_id': above['image_id']}}],
                    'must_not': [
                        {'exists': {'field': 'class_id'}},
                        {'term': {'class_excluded': True}},
                        {'term': {'test_holdout': True}},
                    ],
                }
            },
        },
    )['hits']['hits']
    assert len(hits) == labelled
    for hit in hits:
        resp = api_client.put(
            f'/crops/{hit["_id"]}/label',
            json={'class_id': class_ids[CLASS_NAMES[3]], 'label_source': 'human'},
        )
        assert resp.status_code == 200, resp.text

    def swapped() -> list[dict[str, Any]] | None:
        served, _ = _all_cards(api_client, sort=SORT)
        now_above = _position(served, above['image_id'])
        now_below = _position(served, below['image_id'])
        return served if now_below < now_above else None

    served = wait_until(swapped, timeout=DRAIN_DEADLINE_S)
    assert served, 'the stored-count order never swapped the two images within the drain deadline'
    # Only `above` lost items, so `below` moves up exactly one place, into `above`'s old spot.
    assert _position(served, below['image_id']) == start_above
    counts = [c['counts']['unlabeled'] for c in served]
    assert counts == sorted(counts, reverse=True)
