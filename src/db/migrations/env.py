"""Alembic environment: synchronous, on the connection the runner already holds.

The runner is a one-shot script that keeps a session advisory lock on one connection for the
whole run, so the migration runs on that same connection (Alembic cookbook, "Sharing a
Connection across one or more programmatic migration commands") instead of opening its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alembic import context

import src.db.models  # noqa: F401  # registers every table on Base.metadata
from src.db.base import Base


if TYPE_CHECKING:
    from sqlalchemy.engine import Connection


config = context.config
target_metadata = Base.metadata


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get('connection')
    if connection is None:
        msg = (
            'no database connection: run migrations through scripts/db/migrate.py, which passes '
            'its locked owner connection in config.attributes["connection"]'
        )
        raise RuntimeError(msg)
    _run(connection)


if context.is_offline_mode():
    msg = 'offline (SQL script) migrations are not supported'
    raise RuntimeError(msg)
run_migrations_online()
