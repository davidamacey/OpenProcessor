"""The test PostgreSQL for the ``db`` tier.

One server per test session, shared by every xdist worker: the controller starts a
testcontainers PostgreSQL in ``pytest_configure`` (only when the ``-m`` expression selects
``db``) and hands its address to the workers; a CI service container plays the same part
through ``OP_TEST_DATABASE_URL``. Each worker migrates its own template database once, with
the same runner production uses (roles, advisory lock, Alembic, grants), and each test module
gets a fresh copy of it.

When a database is needed and none can be reached the fixture raises
:class:`DatabaseUnavailable`; it never skips, so a green ``-m db`` run proves something ran.
"""

from __future__ import annotations

import itertools
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session

from scripts.db import migrate
from src.config.database import DatabaseSettings


if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.engine import Engine

# The image and digest of the Postgres service in docker-compose.yml.
POSTGRES_IMAGE = (
    'postgres:18.6-trixie@sha256:bd5dfa494a5a259704365813dbfa5fa6b3e69be7348d43b547f540c730478583'
)
URL_ENV = 'OP_TEST_DATABASE_URL'
_PASSWORDS = {
    'app': 'op-test-app-password',
    'owner': 'op-test-owner-password',
    'super': 'op-test-super-password',
}
_CONNECT_TIMEOUT_S = 5


class DatabaseUnavailable(RuntimeError):  # noqa: N818  # named for the condition, not as an error class
    """The ``db`` tier needs a PostgreSQL and none can be reached."""


@dataclass(frozen=True)
class PgServer:
    host: str
    port: int
    super_user: str
    super_password: str

    def conninfo(self, dbname: str = 'postgres') -> str:
        return psycopg.conninfo.make_conninfo(
            host=self.host,
            port=self.port,
            dbname=dbname,
            user=self.super_user,
            password=self.super_password,
            connect_timeout=_CONNECT_TIMEOUT_S,
        )


class _ContainerHolder:
    """Owns the testcontainers PostgreSQL started on the controller (or in a single process)."""

    def __init__(self) -> None:
        self.container: Any = None
        self.server: PgServer | None = None
        self.error: str | None = None

    def start(self) -> None:
        try:
            from testcontainers.community.postgres import PostgresContainer

            container = PostgresContainer(
                POSTGRES_IMAGE,
                username='op_super',
                password=_PASSWORDS['super'],
                dbname='postgres',
                driver=None,
            )
            container.start()
        except Exception as exc:  # any start failure is reported as unavailable
            self.error = f'cannot start the test PostgreSQL container: {type(exc).__name__}: {exc}'
            return
        self.container = container
        self.server = PgServer(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(5432)),
            super_user='op_super',
            super_password=_PASSWORDS['super'],
        )

    def stop(self) -> None:
        if self.container is not None:
            self.container.stop()
            self.container = None


_holder = _ContainerHolder()


def _selects_db(markexpr: str) -> bool:
    return bool(re.search(r'(?<!not )\bdb\b', markexpr)) or not markexpr.strip()


def _server_from_url(url: str) -> PgServer:
    u = make_url(url.replace('postgresql+psycopg://', 'postgresql://', 1))
    if u.host is None or u.username is None or u.password is None:
        msg = f'{URL_ENV} must carry host, user and password: {u!r}'
        raise DatabaseUnavailable(msg)
    return PgServer(u.host, u.port or 5432, u.username, u.password)


def pytest_configure(config: pytest.Config) -> None:
    if hasattr(config, 'workerinput'):
        return
    if os.environ.get(URL_ENV) or not _selects_db(config.getoption('markexpr') or ''):
        return
    if config.pluginmanager.hasplugin('xdist') and config.getoption('numprocesses', None):
        _holder.start()


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """xdist hook: hand the controller's server (or its start failure) to the worker."""
    if _holder.server is not None:
        node.workerinput['op_db_server'] = _holder.server.__dict__
    if _holder.error is not None:
        node.workerinput['op_db_error'] = _holder.error


def pytest_unconfigure(config: pytest.Config) -> None:
    if not hasattr(config, 'workerinput'):
        _holder.stop()


@pytest.fixture(scope='session')
def pg_server(request: pytest.FixtureRequest) -> PgServer:
    """The shared server, reachable and verified."""
    worker_input = getattr(request.config, 'workerinput', {})
    if worker_input.get('op_db_error'):
        raise DatabaseUnavailable(worker_input['op_db_error'])
    if worker_input.get('op_db_server'):
        server = PgServer(**worker_input['op_db_server'])
    elif os.environ.get(URL_ENV):
        server = _server_from_url(os.environ[URL_ENV])
    else:
        # A single process starts its own server, once, on first use.
        if _holder.server is None and _holder.error is None:
            _holder.start()
        if _holder.server is None:
            raise DatabaseUnavailable(_holder.error or 'no PostgreSQL available')
        server = _holder.server
    try:
        with psycopg.connect(server.conninfo()):
            pass
    except psycopg.OperationalError as exc:
        msg = f'the test PostgreSQL at {server.host}:{server.port} is unreachable: {exc}'
        raise DatabaseUnavailable(msg) from exc
    return server


def _run_as_super(server: PgServer, statement: sql.Composable) -> None:
    with psycopg.connect(server.conninfo(), autocommit=True) as conn:
        conn.execute(statement)


@dataclass(frozen=True)
class PgDatabase:
    """One migrated, throwaway database and the three ways to reach it."""

    server: PgServer
    settings: DatabaseSettings

    def url(self, role: str) -> URL:
        user = self.server.super_user if role == 'super' else self.settings.user(role)
        password = self.server.super_password if role == 'super' else _PASSWORDS[role]
        return URL.create(
            'postgresql+psycopg',
            username=user,
            password=password,
            host=self.server.host,
            port=self.server.port,
            database=self.settings.name,
        )

    def engine(self, role: str = 'app', **kwargs: Any) -> Engine:
        return create_engine(self.url(role), **kwargs)

    def conninfo(self, role: str = 'app') -> str:
        u = self.url(role)
        return psycopg.conninfo.make_conninfo(
            host=u.host, port=u.port, dbname=u.database, user=u.username, password=u.password
        )


def _settings_for(server: PgServer, dbname: str, secrets: Path) -> DatabaseSettings:
    files = {}
    for role, password in _PASSWORDS.items():
        path = secrets / f'{role}_password'
        path.write_text(password)
        files[role] = path
    return replace(
        DatabaseSettings.from_env(),
        host=server.host,
        direct_host=server.host,
        port=server.port,
        name=dbname,
        sslmode='prefer',
        app_password_file=files['app'],
        owner_password_file=files['owner'],
        super_password_file=files['super'],
    )


@pytest.fixture(scope='session')
def _pg_template(pg_server: PgServer) -> Iterator[tuple[PgServer, str, Path]]:
    worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
    template = f'op_template_{worker}'
    with tempfile.TemporaryDirectory(prefix='op-pg-secrets-') as raw:
        secrets = Path(raw)
        _run_as_super(
            pg_server,
            sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(template)),
        )
        _run_as_super(pg_server, sql.SQL('CREATE DATABASE {}').format(sql.Identifier(template)))
        migrate.upgrade(
            _settings_for(pg_server, template, secrets), super_user=pg_server.super_user
        )
        yield pg_server, template, secrets
        _run_as_super(
            pg_server,
            sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(template)),
        )


_counter = itertools.count()


@pytest.fixture(scope='module')
def pg(_pg_template: tuple[PgServer, str, Path]) -> Iterator[PgDatabase]:
    """A fresh copy of the migrated template for one test module."""
    server, template, secrets = _pg_template
    worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
    name = f'op_t_{worker}_{next(_counter)}'
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
            server,
            sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(name)),
        )


@pytest.fixture
def two_sessions(pg: PgDatabase) -> Iterator[tuple[Session, Session]]:
    """Two sessions on two connections of the application role, for race tests."""
    engine = pg.engine('app')
    first, second = Session(engine), Session(engine)
    try:
        yield first, second
    finally:
        first.close()
        second.close()
        engine.dispose()
