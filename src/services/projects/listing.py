"""Order and page the project list in SQL: one sort id, one keyset cursor.

``name:asc`` and ``updated_at:desc`` are keyset pages straight from Postgres
(``ORDER BY <value>, slug`` with a ``(value, slug)`` row-comparison predicate and ``LIMIT``; see
``repo.list_page``). The cursor format is the shared one in ``services.curation.list_cursor``.

``items:desc`` has no stored count: item counts live in OpenSearch, one query per project. It
therefore ranks only the ``ITEMS_SORT_CAP`` most recently updated matches and serves a single
page; the response carries ``sort_cap`` so a client can say so. It is never run over an
unbounded set.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from src.db.repos import projects as repo
from src.db.session import deployment_tx
from src.services.curation.list_cursor import CursorError, decode_cursor, encode_cursor
from src.services.projects.records import record_from_row


# Statuses a list shows by default (archived is opt-in) and statuses that can be opened.
ALWAYS_LISTED = ('active', 'building', 'failed', 'deleting')
SELECTABLE = ('active', 'archived')
#: Filter options in display order; the labels come from ``ProjectLabels``.
FILTER_STATUSES = ('active', 'archived', 'building', 'failed', 'deleting')

#: Page size when a request sends none, and the most one page may hold (both served).
DEFAULT_LIMIT = 50
MAX_LIMIT = 200

ITEMS_SORT = 'items:desc'
ITEMS_SORT_CAP = 200


@dataclass(frozen=True)
class ProjectSort:
    id: str
    label: str
    keyset: str  # the SQL order the page is read in


SORTS: tuple[ProjectSort, ...] = (
    ProjectSort('name:asc', 'Name', 'name:asc'),
    ProjectSort('updated_at:desc', 'Recently updated', 'updated_at:desc'),
    ProjectSort(ITEMS_SORT, 'Most items', 'updated_at:desc'),
)
DEFAULT_SORT = SORTS[0].id


def sort_by_id(sort_id: str) -> ProjectSort:
    for option in SORTS:
        if option.id == sort_id:
            return option
    msg = f'unknown sort {sort_id!r}; one of {[o.id for o in SORTS]}'
    raise CursorError(msg)


class UnknownStatusError(ValueError):
    """A ``status`` filter value that is not one of :data:`FILTER_STATUSES`."""


def listed_statuses(
    *, include_archived: bool, selectable: bool, status: Collection[str] | None
) -> tuple[str, ...]:
    """The statuses a request can match: what the list shows, narrowed by the picker flag and filter.

    A ``status`` value outside :data:`FILTER_STATUSES` raises :class:`UnknownStatusError`; naming
    ``archived`` asks for the archived projects, so it needs no ``include_archived`` beside it.
    """
    wanted = set(status or ())
    if unknown := sorted(wanted - set(FILTER_STATUSES)):
        msg = f'unknown status {unknown[0]!r}; one of {list(FILTER_STATUSES)}'
        raise UnknownStatusError(msg)
    allowed = (*ALWAYS_LISTED, *(('archived',) if include_archived or 'archived' in wanted else ()))
    if selectable:
        allowed = tuple(s for s in allowed if s in SELECTABLE)
    if wanted:
        allowed = tuple(s for s in allowed if s in wanted)
    return allowed


def _after(sort: ProjectSort, cursor: str | None) -> tuple[Any, str] | None:
    if cursor is None:
        return None
    after = decode_cursor(cursor, sort_id=sort.id)
    if len(after) != 2:
        msg = 'cursor is not one this route served'
        raise CursorError(msg)
    value, slug = after
    if sort.keyset == 'updated_at:desc':
        try:
            value = dt.datetime.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            msg = 'cursor is not one this route served'
            raise CursorError(msg) from exc
    return value, str(slug)


async def fetch_page(
    session: Any,
    *,
    statuses: Collection[str],
    q: str | None,
    sort: ProjectSort,
    cursor: str | None,
    limit: int | None,
) -> tuple[list[Any], str | None, int]:
    """``(records, next_cursor, total)``. ``total`` counts every match; ``next_cursor`` is ``None`` on
    the last page and always for the item-count sort."""
    capped = sort.id == ITEMS_SORT
    if capped and cursor is not None:
        msg = 'cursor belongs to a different sort'
        raise CursorError(msg)
    after = _after(sort, cursor)
    page_size = ITEMS_SORT_CAP if capped else min(limit or DEFAULT_LIMIT, MAX_LIMIT)
    async with deployment_tx(session, 'list projects'):
        total = await repo.count_matching(session, statuses=statuses, q=q)
        rows = await repo.list_page(
            session,
            statuses=statuses,
            q=q,
            sort=sort.keyset,
            after=after,
            limit=page_size + 1,
        )
    more = len(rows) > page_size
    rows = rows[:page_size]
    records = [record_from_row(r) for r in rows]
    next_cursor = None
    if more and not capped and rows:
        last = rows[-1]
        value = last.updated_at.isoformat() if sort.keyset == 'updated_at:desc' else last.slug
        next_cursor = encode_cursor(sort.id, [value, last.slug])
    return records, next_cursor, total


def rank_by_items(records: list[Any], counts: Mapping[str, Any], limit: int | None) -> list[Any]:
    """``records`` by item count, highest first (unknown last), slug ascending within a tie."""

    def key(record: Any) -> tuple[int, str]:
        count = counts.get(record.slug)
        items = -1 if count is None or count.items is None else int(count.items)
        return (-items, record.slug)

    ranked = sorted(records, key=key)
    return ranked[: min(limit or DEFAULT_LIMIT, MAX_LIMIT)]
