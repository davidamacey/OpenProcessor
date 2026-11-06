"""Startup checks every process runs before it serves or works: dependencies, then schema head.

``wait_for_dependencies`` polls the required services with a fixed backoff for a bounded time
and then refuses with ``dependency_unavailable: <name>``; there is no degraded mode and no
switch for one. ``check_control_plane`` compares the database revision with the head bundled in
this image and refuses with ``schema_not_current`` naming both: a new image never runs on an
old schema and an old image never runs on a newer one.

A probe is a callable that returns when its service answers and raises :class:`ProbeFailed`
when it does not. Each probe converts only the errors of its own client, so a bug in a probe
is not mistaken for an outage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

from src.db.engine import one_off_connection


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from sqlalchemy.engine import Connection

# A constant, not a setting: long enough for a database that is still starting, short enough
# that a misconfigured stack fails visibly.
DEPENDENCY_WAIT_BUDGET_S = 60.0
DEPENDENCY_WAIT_INTERVAL_S = 2.0

MIGRATIONS_DIR = Path(__file__).resolve().parent / 'migrations'


class ProbeFailed(Exception):  # noqa: N818  # a probe's verdict, not an error class
    """The probed service did not answer."""


class DependencyUnavailableError(RuntimeError):
    def __init__(self, names: list[str], detail: str):
        super().__init__(f'dependency_unavailable: {", ".join(names)} ({detail})')
        self.names = names


class SchemaNotCurrentError(RuntimeError):
    def __init__(self, database: str | None, code: str):
        super().__init__(
            f'schema_not_current: the database is at {database or "no revision"}, '
            f'this image expects {code}'
        )
        self.database = database
        self.code = code


@dataclass(frozen=True)
class Clock:
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic


def wait_for_dependencies(
    probes: Mapping[str, Callable[[], None]],
    *,
    budget_s: float = DEPENDENCY_WAIT_BUDGET_S,
    interval_s: float = DEPENDENCY_WAIT_INTERVAL_S,
    clock: Clock | None = None,
) -> None:
    """Return when every probe passes; raise :class:`DependencyUnavailableError` when the budget ends."""
    clock = clock or Clock()
    deadline = clock.monotonic() + budget_s
    pending = dict(probes)
    reasons: dict[str, str] = {}
    while True:
        for name, probe in list(pending.items()):
            try:
                probe()
            except ProbeFailed as exc:
                reasons[name] = str(exc)
            else:
                del pending[name]
        if not pending:
            return
        if clock.monotonic() >= deadline:
            names = [n for n in probes if n in pending]
            raise DependencyUnavailableError(names, '; '.join(f'{n}: {reasons[n]}' for n in names))
        clock.sleep(interval_s)


def database_probe() -> None:
    """The database accepts a connection from the application role."""
    try:
        with one_off_connection(direct=False, application_name='op-startup') as conn:
            conn.execute(text('SELECT 1'))
    except (SQLAlchemyError, OSError) as exc:
        raise ProbeFailed(type(exc).__name__) from exc


def code_head() -> str:
    head = ScriptDirectory(str(MIGRATIONS_DIR)).get_current_head()
    if head is None:
        msg = f'no migration revision found in {MIGRATIONS_DIR}'
        raise RuntimeError(msg)
    return head


def database_revision(conn: Connection) -> str | None:
    try:
        return conn.execute(text('SELECT version_num FROM alembic_version')).scalar_one_or_none()
    except ProgrammingError:
        conn.rollback()
        return None


def check_control_plane() -> None:
    """Refuse (:class:`SchemaNotCurrentError`) unless the database is at this image's head."""
    with one_off_connection(direct=False, application_name='op-startup') as conn:
        revision = database_revision(conn)
    head = code_head()
    if revision != head:
        raise SchemaNotCurrentError(revision, head)


__all__ = [
    'DEPENDENCY_WAIT_BUDGET_S',
    'Clock',
    'DependencyUnavailableError',
    'ProbeFailed',
    'SchemaNotCurrentError',
    'check_control_plane',
    'database_probe',
    'wait_for_dependencies',
]
