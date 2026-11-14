"""The totals a cursor-paged list pays for once.

The first page (a request with no cursor, or one minted without counts such as a locate cursor)
counts exactly and runs the hidden-count aggregation; every later page reads both from the cursor
it was handed, so a page costs its one search whatever the depth. The totals are therefore those
of the first page for the whole walk, which is what a scroll through one list reports anyway.
Search ``track_total_hits``: https://docs.opensearch.org/latest/api-reference/search/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.services.curation.hidden_counts import hidden_counts
from src.services.curation.list_cursor import cursor_carry


@dataclass(frozen=True)
class CarriedCounts:
    total: int
    hidden: dict[str, int]

    def as_carry(self) -> dict[str, Any]:
        return {'total': self.total, 'hidden': self.hidden}


def carried_counts(cursor: str | None, *, sort_id: str) -> CarriedCounts | None:
    """The counts ``cursor`` carries, ``None`` when this page has to count.

    Raises :class:`~src.services.curation.list_cursor.CursorError` for a foreign cursor."""
    if cursor is None:
        return None
    carry = cursor_carry(cursor, sort_id=sort_id)
    if carry is None:
        return None
    try:
        hidden = {str(k): int(v) for k, v in dict(carry['hidden']).items()}
        return CarriedCounts(total=int(carry['total']), hidden=hidden)
    except (KeyError, TypeError, ValueError):
        return None


async def page_hidden(
    opensearch: Any,
    carried: CarriedCounts | None,
    filter_clauses: list[dict[str, Any]],
    *,
    include_test: bool,
    include_excluded: bool,
) -> dict[str, int]:
    """The carried hidden counts, or (first page) the aggregation over ``filter_clauses``."""
    if carried is not None:
        return carried.hidden
    return await hidden_counts(
        opensearch, filter_clauses, include_test=include_test, include_excluded=include_excluded
    )


def page_counts(
    carried: CarriedCounts | None, resp: dict[str, Any], hidden: dict[str, int]
) -> CarriedCounts:
    """``carried``, or the exact total of the first page's ``resp`` with its ``hidden`` counts."""
    if carried is not None:
        return carried
    total = ((resp.get('hits') or {}).get('total') or {}).get('value', 0)
    return CarriedCounts(total=int(total), hidden=hidden)
