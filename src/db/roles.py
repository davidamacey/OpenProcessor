"""Database roles, what only a superuser may create, and the grants.

The rule (02a section 1.5): the superuser bootstrap creates the roles and everything the schema
owner may not create; the owner role runs the migrations and grants. The owner has no ``CREATE`` on
the database (PostgreSQL 18, ``GRANT``: ``CREATE`` on a database allows new schemas and, with it,
installing even a trusted extension), so those two live here.

``ensure_roles`` runs as the bootstrap superuser, in the maintenance database, and is idempotent:
it creates the schema owner (not a superuser; ``BYPASSRLS`` only so ``pg_dump`` sees every row),
the application role (not the table owner, ``NOBYPASSRLS``) and the vector role (``op_vector``,
``NOBYPASSRLS``, granted to nobody, 15a section 6.1), and sets the per-role session limits.
``prepare_database`` runs as the superuser in the target database: the owner takes ``public``, the
extensions ``pg_trgm`` and ``vector`` are installed, and the schema ``vec`` is created, owned by the
owner role. ``apply_grants`` runs as the owner after the schema is at head and re-applies the
application role's rights, so a table a new revision adds is covered without a default-privilege
rule, and gives the vector role its schema and its reference to ``projects``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from psycopg import sql

from src.db import vector_ddl


if TYPE_CHECKING:
    from psycopg import Connection

    from src.config.database import DatabaseSettings

# Settings applied with ALTER ROLE so they hold for every new session of the role, wherever it
# connects from. The application role keeps short limits so a runaway request query ends in
# seconds; job workers raise their own at connect time.
APP_ROLE_SETTINGS = {
    'idle_in_transaction_session_timeout': '60s',
    'statement_timeout': '30s',
    'lock_timeout': '5s',
    'timezone': 'UTC',
}
OWNER_ROLE_SETTINGS = {
    'idle_in_transaction_session_timeout': '5min',
    'statement_timeout': '0',
    'lock_timeout': '60s',
    'timezone': 'UTC',
}


class BootstrapError(RuntimeError):
    """A step only the superuser can take cannot be taken: the database is not usable."""


# Both installed by ``prepare_database``. ``pg_trgm`` ships with PostgreSQL (contrib); ``vector``
# (pgvector) ships in the ``pgvector/pgvector`` image the compose files pin (01_package_pins.md).
# A server that offers either one fewer is not a server the record can run on.
_REQUIRED_EXTENSIONS = (
    ('pg_trgm', 'the text-search indexes'),
    ('vector', 'the per-space vector tables'),
)


def _role_exists(conn: Connection, name: str) -> bool:
    return conn.execute('SELECT 1 FROM pg_roles WHERE rolname = %s', (name,)).fetchone() is not None


def _upsert_role(conn: Connection, name: str, password: str, attributes: str) -> None:
    verb = 'ALTER' if _role_exists(conn, name) else 'CREATE'
    conn.execute(
        sql.SQL('{} ROLE {} WITH LOGIN {} PASSWORD {}').format(
            sql.SQL(verb), sql.Identifier(name), sql.SQL(attributes), sql.Literal(password)
        )
    )


def _apply_role_settings(conn: Connection, name: str, settings: dict[str, str]) -> None:
    for key, value in settings.items():
        conn.execute(
            sql.SQL('ALTER ROLE {} SET {} = {}').format(
                sql.Identifier(name), sql.Identifier(key), sql.Literal(value)
            )
        )


def ensure_roles(conn: Connection, settings: DatabaseSettings) -> None:
    """Create or refresh the owner and application roles. ``conn`` is a superuser, autocommit."""
    _upsert_role(
        conn,
        settings.owner_user,
        settings.password('owner'),
        'NOSUPERUSER NOCREATEDB NOCREATEROLE BYPASSRLS',
    )
    _upsert_role(
        conn,
        settings.app_user,
        settings.password('app'),
        'NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS',
    )
    _upsert_role(
        conn,
        settings.user('vector'),
        settings.password('vector'),
        vector_ddl.ROLE_ATTRIBUTES,
    )
    _apply_role_settings(conn, settings.owner_user, OWNER_ROLE_SETTINGS)
    _apply_role_settings(conn, settings.app_user, APP_ROLE_SETTINGS)
    _apply_role_settings(conn, settings.user('vector'), vector_ddl.ROLE_SETTINGS)


def _extension_available(conn: Connection, name: str) -> bool:
    row = conn.execute('SELECT 1 FROM pg_available_extensions WHERE name = %s', (name,)).fetchone()
    return row is not None


def _install_extensions(conn: Connection) -> None:
    for name, needed_for in _REQUIRED_EXTENSIONS:
        if not _extension_available(conn, name):
            raise BootstrapError(
                f'the PostgreSQL server does not offer the extension {name}, which {needed_for} need: '
                'run the pinned pgvector/pgvector image (docker-compose.yml, postgres service)'
            )
        conn.execute(
            sql.SQL('CREATE EXTENSION IF NOT EXISTS {} WITH SCHEMA public').format(
                sql.Identifier(name)
            )
        )


def prepare_database(conn: Connection, settings: DatabaseSettings) -> None:
    """Take the steps the owner may not. Run as the superuser, in the target database.

    ``CREATE SCHEMA`` and ``CREATE EXTENSION`` need ``CREATE`` on the database
    (https://www.postgresql.org/docs/18/sql-createschema.html,
    https://www.postgresql.org/docs/18/sql-createextension.html), which the owner role does not
    have. ``vec`` is owned by the owner role (``AUTHORIZATION``), so the owner can grant on it and
    drop it; the vector role gets ``USAGE, CREATE`` from ``apply_grants``.
    """
    owner = sql.Identifier(settings.owner_user)
    conn.execute(sql.SQL('ALTER SCHEMA public OWNER TO {}').format(owner))
    for role in (settings.app_user, settings.user('vector')):
        conn.execute(
            sql.SQL('GRANT CONNECT ON DATABASE {} TO {}').format(
                sql.Identifier(settings.name), sql.Identifier(role)
            )
        )
    _install_extensions(conn)
    schema = sql.Identifier(vector_ddl.SCHEMA)
    conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {} AUTHORIZATION {}').format(schema, owner))
    conn.execute(sql.SQL('ALTER SCHEMA {} OWNER TO {}').format(schema, owner))


def apply_grants(conn: Connection, settings: DatabaseSettings) -> None:
    """Give the application role data rights and nothing else. Run as the owner."""
    app = sql.Identifier(settings.app_user)
    for statement in (
        'GRANT USAGE ON SCHEMA public TO {}',
        # the per-space vector tables are made in this schema by op_vector, which grants the table rights
        'GRANT USAGE ON SCHEMA vec TO {}',
        'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}',
        'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}',
        'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO {}',
        # The schema check reads the revision; only a migration may change it.
        'REVOKE INSERT, UPDATE, DELETE ON alembic_version FROM {}',
    ):
        conn.execute(sql.SQL(statement).format(app))
    # The vector role makes tables in ``vec`` and references ``projects``; the owner owns both.
    for statement in vector_ddl.role_grants():
        conn.execute(sql.SQL(statement))
