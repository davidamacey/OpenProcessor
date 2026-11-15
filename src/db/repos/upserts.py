"""``INSERT ... ON CONFLICT DO UPDATE ... WHERE row IS DISTINCT FROM`` for the record tables.

The one statement shape every result and state writer uses (15a section 1, "Updated rows"): a row
that already holds the same values writes no tuple, so a re-run or a fold that changes nothing leaves
the table, its indexes and its HOT chain alone. PostgreSQL 18 INSERT: "ON CONFLICT DO UPDATE ... The
optional WHERE clause ... only rows for which the condition returns true will be updated"
(https://www.postgresql.org/docs/18/sql-insert.html); ``IS DISTINCT FROM`` treats NULLs as equal
(https://www.postgresql.org/docs/18/functions-comparison.html). Multi-row ``VALUES`` in chunks that stay
under the 65,535-parameter limit of the wire protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import ARRAY, any_, bindparam, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.elements import ColumnElement

PARAMETER_BUDGET = 30_000
Row = Mapping[str, Any]


class RowShapeError(ValueError):
    """Rows of one statement that do not share their columns, or lack a key column."""


def any_of(column: Any, values: Iterable[Any]) -> ColumnElement[bool]:
    """``column = ANY(:array)``: one parameter however many values (an ``IN`` list would use one
    parameter per value and meet the 65,535 limit)."""
    return cast(
        'ColumnElement[bool]',
        column == any_(bindparam(None, list(values), type_=ARRAY(column.type))),
    )


def _distinct(cols: Sequence[Any], new: Sequence[Any]) -> ColumnElement[bool]:
    if len(cols) == 1:
        return cast('ColumnElement[bool]', cols[0].is_distinct_from(new[0]))
    return tuple_(*cols).is_distinct_from(tuple_(*new))


def dedupe(rows: Sequence[Row], key: Sequence[str]) -> list[Row]:
    """The last row per key: ``ON CONFLICT DO UPDATE`` cannot touch one row twice in a statement."""
    last: dict[tuple[Any, ...], Row] = {}
    for row in rows:
        last[tuple(row[k] for k in key)] = row
    return list(last.values())


async def upsert_distinct(
    session: AsyncSession,
    model: Any,
    rows: Sequence[Row],
    *,
    key: Sequence[str],
    no_compare: Sequence[str] = (),
    set_extra: Mapping[str, Any] | None = None,
    guard: Callable[[Any, Any], ColumnElement[bool]] | None = None,
) -> int:
    """Insert the rows; where the key exists, update it only when a value differs.

    ``no_compare`` columns are written with the row but do not decide whether it differs (a
    timestamp that moves only with another column). ``set_extra`` adds assignments to the update
    only (``{'updated_at': func.now()}``). ``guard(table, excluded)`` adds a condition the update
    must also meet. Returns the rows inserted or updated.
    """
    if not rows:
        return 0
    columns = list(rows[0])
    missing = [k for k in key if k not in columns]
    if missing:
        raise RowShapeError(f'{model.__tablename__}: rows lack key columns {missing}')
    if any(list(r) != columns for r in rows):
        raise RowShapeError(f'{model.__tablename__}: rows do not share their columns')
    table = model.__table__
    body = [c for c in columns if c not in key]
    compared = [c for c in body if c not in no_compare]
    written = 0
    chunk = max(1, PARAMETER_BUDGET // len(columns))
    for start in range(0, len(rows), chunk):
        insert = pg_insert(table).values(
            [dict(r) for r in dedupe(rows[start : start + chunk], key)]
        )
        if not body:
            statement = insert.on_conflict_do_nothing(index_elements=list(key))
        else:
            where = _distinct(
                [table.c[c] for c in compared], [insert.excluded[c] for c in compared]
            )
            if guard is not None:
                where = where & guard(table, insert.excluded)
            assignments: dict[str, Any] = {c: insert.excluded[c] for c in body}
            assignments.update(set_extra or {})
            statement = insert.on_conflict_do_update(
                index_elements=list(key), set_=assignments, where=where
            )
        result = await session.execute(statement)
        written += int(result.rowcount)  # type: ignore[attr-defined]  # a DML result
    return written
