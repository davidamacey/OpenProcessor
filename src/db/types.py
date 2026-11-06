"""Column types and server-default expressions shared by every model."""

from __future__ import annotations

from typing import Any

from sqlalchemy import CheckConstraint, text
from sqlalchemy.types import UserDefinedType


UUIDV7 = text('uuidv7()')
NOW = text('now()')
EMPTY_OBJECT = text("'{}'::jsonb")
EMPTY_ARRAY = text("'{}'")


class Xid8(UserDefinedType[int]):
    """PostgreSQL ``xid8``: a 64-bit transaction id that does not wrap, used as the commit-safe event cursor."""

    cache_ok = True

    def get_col_spec(self, **_kw: Any) -> str:
        return 'xid8'


def check(name: str, expression: str) -> CheckConstraint:
    """A named ``CHECK``; the naming convention prefixes the table."""
    return CheckConstraint(expression, name=name)


def in_list(column: str, values: tuple[str, ...]) -> str:
    """``column IN ('a', 'b')`` for a closed set whose one Python source is the caller's tuple."""
    quoted = ', '.join("'" + v.replace("'", "''") + "'" for v in values)
    return f'{column} IN ({quoted})'
