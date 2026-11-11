"""The one opaque `search_after` cursor every cursored list route uses."""

from __future__ import annotations

import pytest

from src.services.curation.list_cursor import (
    CursorError,
    decode_cursor,
    encode_cursor,
    page_body,
    split_page,
)


def test_round_trip_keeps_the_sort_values() -> None:
    cursor = encode_cursor('score:desc', [0.5, 'crop-1', None])
    assert decode_cursor(cursor, sort_id='score:desc') == [0.5, 'crop-1', None]


@pytest.mark.parametrize('garbage', ['', 'bm9wZQ==', '!!!', 'e30=', 'eyJzb3J0IjoieCJ9'])
def test_garbage_is_refused(garbage: str) -> None:
    with pytest.raises(CursorError):
        decode_cursor(garbage, sort_id='x')


def test_a_cursor_of_another_sort_is_refused() -> None:
    cursor = encode_cursor('score:desc', [1])
    with pytest.raises(CursorError, match='different sort'):
        decode_cursor(cursor, sort_id='score:asc')


def test_page_body_asks_for_one_extra_and_ends_on_the_tiebreak() -> None:
    body = page_body(
        {'query': {'match_all': {}}},
        sort=[{'score': {'order': 'desc'}}],
        tiebreak='crop_id',
        cursor=None,
        sort_id='score:desc',
        limit=2,
    )
    assert body['size'] == 3
    assert body['sort'] == [{'score': {'order': 'desc'}}, {'crop_id': {'order': 'asc'}}]
    assert 'search_after' not in body
    assert body['query'] == {'match_all': {}}


def test_page_body_resumes_after_the_cursor() -> None:
    cursor = encode_cursor('s', [3, 'c9'])
    body = page_body({}, sort=[], tiebreak='crop_id', cursor=cursor, sort_id='s', limit=5)
    assert body['search_after'] == [3, 'c9']


def test_split_page_gives_a_cursor_only_when_more_remain() -> None:
    hits = [{'sort': [i, f'c{i}']} for i in range(3)]
    kept, nxt = split_page(hits, sort_id='s', limit=2)
    assert kept == hits[:2]
    assert nxt is not None
    assert decode_cursor(nxt, sort_id='s') == [1, 'c1']
    assert split_page(hits[:2], sort_id='s', limit=2) == (hits[:2], None)
    assert split_page([], sort_id='s', limit=2) == ([], None)


def test_page_body_keeps_a_sort_that_already_ends_on_the_tiebreak() -> None:
    sort = [{'score': {'order': 'desc'}}, {'crop_id': {'order': 'asc'}}]
    body = page_body({}, sort=sort, tiebreak='crop_id', cursor=None, sort_id='s', limit=1)
    assert body['sort'] == sort
