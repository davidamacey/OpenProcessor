"""A session that records the statements it is given and answers from a script: the offline tests of
the repositories read the SQL they send, not a database's answer."""

from __future__ import annotations

from typing import Any

from sqlalchemy.dialects import postgresql


class FakeResult:
    def __init__(self, rows: list[Any] | None = None, rowcount: int = 0) -> None:
        self._rows = rows or []
        self.rowcount = rowcount or len(self._rows)

    def __iter__(self) -> Any:
        return iter(self._rows)

    def all(self) -> list[Any]:
        return list(self._rows)

    def scalars(self) -> Any:
        return iter(self._rows)

    def scalar_one_or_none(self) -> Any:
        return self._rows[0] if self._rows else None

    def scalar_one(self) -> Any:
        return self._rows[0]


class FakeSession:
    def __init__(self, *answers: FakeResult) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self._answers = list(answers)

    async def execute(self, statement: Any, params: Any = None) -> FakeResult:
        self.calls.append((statement, params))
        return self._answers.pop(0) if self._answers else FakeResult()

    def sql(self, n: int = 0) -> str:
        statement = self.calls[n][0]
        return ' '.join(str(statement.compile(dialect=postgresql.dialect())).split())


class Row:
    """A result row: attribute access, as SQLAlchemy's ``Row``."""

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)
