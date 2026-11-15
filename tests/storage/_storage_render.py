"""Render a revision's ``upgrade()`` and ``downgrade()`` to SQL without a database.

Alembic's documented way to generate a script without ``env.py``: a ``MigrationContext`` in
``as_sql`` mode under ``Operations.context`` (https://alembic.sqlalchemy.org/en/latest/api/runtime.html,
"Running programmatically"). ``src/db/migrations/env.py`` refuses offline mode by design, so the tests
drive the revision module directly.
"""

from __future__ import annotations

import importlib
import io
import re
from functools import cache
from typing import TYPE_CHECKING

from alembic.migration import MigrationContext
from alembic.operations import Operations


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

REVISION = '0010_storage_record'


def revision_module() -> ModuleType:
    return importlib.import_module(f'src.db.migrations.versions.{REVISION}')


def _render(fn: Callable[[], None]) -> str:
    buffer = io.StringIO()
    context = MigrationContext.configure(
        dialect_name='postgresql', opts={'as_sql': True, 'output_buffer': buffer}
    )
    with Operations.context(context):
        fn()
    return buffer.getvalue()


@cache
def upgrade_sql() -> str:
    return _render(revision_module().upgrade)


@cache
def downgrade_sql() -> str:
    return _render(revision_module().downgrade)


def statements(sql: str) -> list[str]:
    """The script split on its statement terminators (a ``;`` at a line end outside a ``$$`` body)."""
    out: list[str] = []
    current: list[str] = []
    in_body = False
    for line in sql.splitlines():
        current.append(line)
        in_body ^= line.count('$$') % 2 == 1
        if not in_body and line.rstrip().endswith(';'):
            out.append('\n'.join(current).strip())
            current = []
    return [s for s in out if s]


def create_table_columns(statement: str) -> list[str]:
    """Column names of one ``CREATE TABLE`` statement (constraint lines are not columns)."""
    body = statement[statement.index('(') + 1 : statement.rindex(')')]
    depth = 0
    parts: list[str] = []
    start = 0
    for i, ch in enumerate(body):
        depth += ch == '('
        depth -= ch == ')'
        if ch == ',' and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    names = []
    for part in parts:
        first = part.split(None, 1)[0] if part.strip() else ''
        if first.upper() not in {'CONSTRAINT', 'PRIMARY', 'UNIQUE', 'CHECK', 'FOREIGN'}:
            names.append(first)
    return names


def create_tables(sql: str) -> dict[str, str]:
    found = {}
    for s in statements(sql):
        m = re.match(r'CREATE TABLE (\w+) \(', s)
        if m:
            found[m.group(1)] = s
    return found
