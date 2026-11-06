"""Database roles and their grants.

``ensure_roles`` runs as the bootstrap superuser and is idempotent: it creates the schema owner
(not a superuser; ``BYPASSRLS`` only so ``pg_dump`` sees every row) and the application role
(not the table owner, ``NOBYPASSRLS``), and sets the per-role session limits. ``apply_grants``
runs as the owner after the schema is at head and re-applies the application role's rights, so
a table a new revision adds is covered without a default-privilege rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from psycopg import sql


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
    _apply_role_settings(conn, settings.owner_user, OWNER_ROLE_SETTINGS)
    _apply_role_settings(conn, settings.app_user, APP_ROLE_SETTINGS)


def prepare_database(conn: Connection, settings: DatabaseSettings) -> None:
    """Let the owner create objects in ``public``. Run as the superuser, in the target database."""
    conn.execute(
        sql.SQL('ALTER SCHEMA public OWNER TO {}').format(sql.Identifier(settings.owner_user))
    )
    conn.execute(
        sql.SQL('GRANT CONNECT ON DATABASE {} TO {}').format(
            sql.Identifier(settings.name), sql.Identifier(settings.app_user)
        )
    )


def apply_grants(conn: Connection, settings: DatabaseSettings) -> None:
    """Give the application role data rights and nothing else. Run as the owner."""
    app = sql.Identifier(settings.app_user)
    for statement in (
        'GRANT USAGE ON SCHEMA public TO {}',
        'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}',
        'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}',
        'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO {}',
        # The schema check reads the revision; only a migration may change it.
        'REVOKE INSERT, UPDATE, DELETE ON alembic_version FROM {}',
    ):
        conn.execute(sql.SQL(statement).format(app))
