"""One opaque cursor for every list route that pages with `search_after`.

The cursor is URL-safe base64 of `{"sort": <sort id>, "after": <sort values>}`. It is refused when
the sort differs, so a cursor from one order never resumes another. Every sort ends on a unique
tiebreak field so `search_after` is total. A point-in-time is not used: the project guard keeps PIT
ids per process, and a PIT handed to a browser would be refused by another worker.

* search_after: https://docs.opensearch.org/latest/search-plugins/searching-data/paginate/
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any


class CursorError(ValueError):
    """The cursor is garbage or belongs to a different sort."""


def encode_cursor(sort_id: str, after: list[Any]) -> str:
    raw = json.dumps({'sort': sort_id, 'after': after}, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str, *, sort_id: str) -> list[Any]:
    try:
        body = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        after = body['after']
        cursor_sort = body['sort']
    except (binascii.Error, ValueError, KeyError, TypeError) as exc:
        msg = 'cursor is not one this route served'
        raise CursorError(msg) from exc
    if cursor_sort != sort_id or not isinstance(after, list):
        msg = 'cursor belongs to a different sort'
        raise CursorError(msg)
    return after


def page_body(
    body: dict[str, Any],
    *,
    sort: list[dict[str, Any]],
    tiebreak: str,
    cursor: str | None,
    sort_id: str,
    limit: int,
) -> dict[str, Any]:
    """``body`` with ``limit + 1`` rows asked for (the extra one says whether a next page exists)."""
    out = {**body, 'size': limit + 1, 'sort': [*sort, {tiebreak: {'order': 'asc'}}]}
    if cursor is not None:
        out['search_after'] = decode_cursor(cursor, sort_id=sort_id)
    return out


def split_page(
    hits: list[dict[str, Any]], *, sort_id: str, limit: int
) -> tuple[list[dict[str, Any]], str | None]:
    """The hits to serve and the cursor of the next page, or ``None`` on the last page."""
    kept = hits[:limit]
    if len(hits) > limit and kept:
        return kept, encode_cursor(sort_id, kept[-1]['sort'])
    return kept, None
