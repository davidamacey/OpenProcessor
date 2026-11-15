"""The superuser bootstrap and the owner's grants, run against a recording connection.

15a section 6.1 and 02a section 1.5: the roles and everything the schema owner may not create
(``CREATE`` on the database is the owner's missing right) are made by the bootstrap, as the
superuser; the owner role only grants and creates inside what it owns. No server is involved: what a
server must answer (that PostgreSQL accepts each statement) is in the ``db`` tests of this change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import pytest

from src.config.database import DatabaseSettings
from src.db import roles


if TYPE_CHECKING:
    from pathlib import Path

    from psycopg import Connection


class RecordingConnection:
    """Stands for a psycopg connection: keeps every statement, answers the two catalog lookups."""

    def __init__(
        self, *, existing_roles: set[str] | None = None, available: set[str] | None = None
    ):
        self.statements: list[str] = []
        self._roles = set(existing_roles or ())
        self._available = set(available or ())

    @property
    def connection(self) -> Connection:
        return cast('Connection', self)

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else query.as_string()
        self.statements.append(text)
        row = None
        if 'FROM pg_roles' in text:
            row = (1,) if params and params[0] in self._roles else None
        elif 'FROM pg_available_extensions' in text:
            row = (1,) if params and params[0] in self._available else None
        return _Result(row)


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


@pytest.fixture
def settings(tmp_path: Path) -> DatabaseSettings:
    files = {}
    for name in ('app', 'owner', 'super', 'vector'):
        path = tmp_path / f'{name}_password'
        path.write_text(f'pw-{name}')
        files[name] = path
    return DatabaseSettings(
        host='postgres',
        direct_host='postgres',
        port=5432,
        name='openprocessor',
        app_user='op_app',
        owner_user='op_owner',
        app_password_file=files['app'],
        owner_password_file=files['owner'],
        super_password_file=files['super'],
        vector_password_file=files['vector'],
        sslmode='prefer',
        pooler='none',
        api_replicas=1,
        api_connections_total=128,
    )


def _joined(conn: RecordingConnection) -> str:
    return '\n'.join(conn.statements)


def test_the_bootstrap_creates_the_vector_role_without_super_or_bypass_rights(settings) -> None:
    conn = RecordingConnection()
    roles.ensure_roles(conn.connection, settings)
    create = next(s for s in conn.statements if 'ROLE "op_vector"' in s and s.startswith('CREATE'))
    assert 'LOGIN' in create
    assert 'NOSUPERUSER' in create
    assert 'NOBYPASSRLS' in create
    assert "PASSWORD 'pw-vector'" in create
    # its own timeouts are on the role: SET ROLE would not apply them (15a section 6.1)
    assert 'ALTER ROLE "op_vector" SET "statement_timeout" = \'0\'' in _joined(conn)
    assert 'ALTER ROLE "op_vector" SET "lock_timeout" = \'0\'' in _joined(conn)


def test_a_rerun_alters_an_existing_vector_role(settings) -> None:
    conn = RecordingConnection(existing_roles={'op_vector'})
    roles.ensure_roles(conn.connection, settings)
    assert any(s.startswith('ALTER ROLE "op_vector" WITH LOGIN') for s in conn.statements)
    assert not any(s.startswith('CREATE ROLE "op_vector"') for s in conn.statements)


def test_the_vector_role_is_never_granted_to_another_role(settings) -> None:
    conn = RecordingConnection(available={'pg_trgm'})
    roles.ensure_roles(conn.connection, settings)
    roles.prepare_database(conn.connection, settings)
    roles.apply_grants(conn.connection, settings)
    for statement in conn.statements:
        assert not statement.startswith('GRANT "op_vector" TO'), statement
        assert not statement.startswith('GRANT op_vector TO'), statement


def test_the_database_step_makes_what_the_owner_may_not_create(settings) -> None:
    conn = RecordingConnection(available={'vector', 'pg_trgm'})
    roles.prepare_database(conn.connection, settings)
    sql = _joined(conn)
    # CREATE SCHEMA needs CREATE on the database, and CREATE EXTENSION does too even for a trusted one
    assert 'CREATE SCHEMA IF NOT EXISTS "vec" AUTHORIZATION "op_owner"' in sql
    assert 'CREATE EXTENSION IF NOT EXISTS "pg_trgm"' in sql
    assert 'CREATE EXTENSION IF NOT EXISTS "vector"' in sql
    assert 'GRANT CONNECT ON DATABASE "openprocessor" TO "op_vector"' in sql


def test_a_missing_vector_package_is_a_warning_and_a_missing_trigram_module_is_not(
    settings, caplog: pytest.LogCaptureFixture
) -> None:
    conn = RecordingConnection(available={'pg_trgm'})
    with caplog.at_level(logging.WARNING):
        roles.prepare_database(conn.connection, settings)
    sql = _joined(conn)
    assert 'CREATE EXTENSION IF NOT EXISTS "vector"' not in sql
    assert 'CREATE EXTENSION IF NOT EXISTS "pg_trgm"' in sql
    assert any('vector' in r.getMessage() and 'pgvector' in r.getMessage() for r in caplog.records)

    broken = RecordingConnection(available={'vector'})
    with pytest.raises(roles.BootstrapError, match='pg_trgm'):
        roles.prepare_database(broken.connection, settings)


def test_the_owner_grants_the_vector_role_its_schema_and_the_project_reference(settings) -> None:
    conn = RecordingConnection()
    roles.apply_grants(conn.connection, settings)
    sql = _joined(conn)
    assert 'GRANT USAGE, CREATE ON SCHEMA vec TO op_vector' in sql
    assert 'GRANT REFERENCES (id) ON projects TO op_vector' in sql
    # the application role may use the schema and nothing more: it never creates a vector table
    assert 'GRANT USAGE ON SCHEMA vec TO "op_app"' in sql
    assert not any('CREATE' in s and 'vec' in s and 'op_app' in s for s in conn.statements)
