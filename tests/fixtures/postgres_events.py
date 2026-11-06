"""A PostgreSQL server of its own for the tests that depend on the event cursor.

The cursor emits only rows whose transaction is older than every running one
(``pg_snapshot_xmin``), and that horizon is cluster-wide: an open transaction in *any* database of
the server delays visibility in all of them. Under xdist the shared test server runs other
workers' transactions, so cursor tests there are timing-dependent. They get a separate server
(not a separate database) per xdist worker: a testcontainers PostgreSQL, or the server named by
``OP_TEST_EVENTS_DATABASE_URL``.

Production behaves the same way by design (02b section 6.3): delivery waits for the oldest open
transaction in the cluster, which the role statement and idle-in-transaction timeouts bound (D-72).

A test module opts in by overriding the ``pg`` fixture::

    @pytest.fixture(scope='module')
    def pg(pg_events):
        return pg_events
"""

from __future__ import annotations

import itertools
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fixtures.postgres import (
    DatabaseUnavailable,
    PgDatabase,
    PgServer,
    _ContainerHolder,
    _run_as_super,
    _server_from_url,
    _settings_for,
)
from psycopg import sql

from scripts.db import migrate


if TYPE_CHECKING:
    from collections.abc import Iterator

EVENTS_URL_ENV = 'OP_TEST_EVENTS_DATABASE_URL'
_counter = itertools.count()


@pytest.fixture(scope='session')
def pg_events_server() -> Iterator[PgServer]:
    """This worker's own server, never the one the rest of the db tier shares."""
    if url := os.environ.get(EVENTS_URL_ENV):
        yield _server_from_url(url)
        return
    holder = _ContainerHolder()
    holder.start()
    if holder.server is None:
        raise DatabaseUnavailable(holder.error or 'no PostgreSQL available for the event tests')
    try:
        yield holder.server
    finally:
        holder.stop()


@pytest.fixture(scope='session')
def _pg_events_template(pg_events_server: PgServer) -> Iterator[tuple[PgServer, str, Path]]:
    template = 'op_events_template'
    with tempfile.TemporaryDirectory(prefix='op-pg-events-secrets-') as raw:
        secrets = Path(raw)
        _run_as_super(
            pg_events_server,
            sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(template)),
        )
        _run_as_super(
            pg_events_server, sql.SQL('CREATE DATABASE {}').format(sql.Identifier(template))
        )
        migrate.upgrade(
            _settings_for(pg_events_server, template, secrets),
            super_user=pg_events_server.super_user,
        )
        yield pg_events_server, template, secrets


@pytest.fixture(scope='module')
def pg_events(_pg_events_template: tuple[PgServer, str, Path]) -> Iterator[PgDatabase]:
    """A fresh migrated database on the events server for one test module."""
    server, template, secrets = _pg_events_template
    name = f'op_events_{next(_counter)}'
    _run_as_super(
        server,
        sql.SQL('CREATE DATABASE {} TEMPLATE {}').format(
            sql.Identifier(name), sql.Identifier(template)
        ),
    )
    try:
        yield PgDatabase(server, _settings_for(server, name, secrets))
    finally:
        _run_as_super(
            server, sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(name))
        )
