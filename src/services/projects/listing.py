"""Order and page the project list: one sort id, one keyset cursor.

The registry snapshot is the list's source (a Postgres-backed view kept fresh by
``ensure_fresh``), so the keyset is applied to it: rows are ordered by ``(sort value, slug)`` and
the cursor carries the last row's pair, so a page never repeats or skips a project that exists
for the whole walk. The cursor format is the shared one in ``services.curation.list_cursor``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.services.curation.list_cursor import CursorError, decode_cursor, encode_cursor


@dataclass(frozen=True)
class ProjectSort:
    id: str
    label: str
    descending: bool
    value: Callable[[Any, Mapping[str, Any]], Any]


def _items(record: Any, counts: Mapping[str, Any]) -> int:
    count = counts.get(record.slug)
    return -1 if count is None or count.items is None else int(count.items)


SORTS: tuple[ProjectSort, ...] = (
    ProjectSort('name:asc', 'Name', False, lambda r, _c: r.slug),
    ProjectSort('updated_at:desc', 'Recently updated', True, lambda r, _c: r.updated_at),
    ProjectSort('items:desc', 'Most items', True, _items),
)
DEFAULT_SORT = SORTS[0].id


def sort_by_id(sort_id: str) -> ProjectSort:
    for option in SORTS:
        if option.id == sort_id:
            return option
    msg = f'unknown sort {sort_id!r}; one of {[o.id for o in SORTS]}'
    raise CursorError(msg)


def page(
    records: Mapping[str, Any],
    *,
    sort: ProjectSort,
    counts: Mapping[str, Any],
    cursor: str | None,
    limit: int | None,
) -> tuple[list[Any], str | None]:
    """The records of one page in ``sort`` order and the next page's cursor (``None`` at the end)."""
    rows = sorted(records.values(), key=lambda r: r.slug)
    rows.sort(key=lambda r: sort.value(r, counts), reverse=sort.descending)
    if cursor is not None:
        after = decode_cursor(cursor, sort_id=sort.id)
        if len(after) != 2:
            msg = 'cursor is not one this route served'
            raise CursorError(msg)
        value, slug = after
        rows = [r for r in rows if _after(sort.value(r, counts), r.slug, value, slug, sort)]
    if limit is None or len(rows) <= limit:
        return rows, None
    kept = rows[:limit]
    last = kept[-1]
    return kept, encode_cursor(sort.id, [sort.value(last, counts), last.slug])


def _after(value: Any, slug: str, a_value: Any, a_slug: str, sort: ProjectSort) -> bool:
    try:
        if value == a_value:
            return bool(slug > a_slug)
        return bool(value < a_value if sort.descending else value > a_value)
    except TypeError as exc:
        msg = 'cursor is not one this route served'
        raise CursorError(msg) from exc
