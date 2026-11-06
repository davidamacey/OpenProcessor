#!/usr/bin/env python3
"""The one place that changes the schema: ``upgrade``, ``current`` and ``check``.

``upgrade`` is what the one-shot ``db-migrate`` service runs. As the bootstrap superuser it
creates the owner and application roles; then, as the owner on the direct host (never through
a pooler: the lock below is a session lock), it takes ``pg_advisory_lock(2, 0)`` for the whole
run, upgrades to head on that same connection, re-applies the application role's grants and
unlocks. Two runners started together serialize and the second finds the schema at head.

After the schema is at head the same step syncs the embedding model catalog, creates the
``default`` project of a fresh install and runs the one OpenSearch migration
(``scripts/opensearch/migrate.py``), so no API worker creates or migrates an index.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, pool, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from scripts.opensearch.migrate import run as opensearch_run
from src.config.advisory_locks import MIGRATE
from src.config.database import SUPER_USER, DatabaseSettings
from src.db import roles
from src.db.engine import one_off_connection, run_with_engine
from src.services.embedding_spaces.catalog import sync_catalog


if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATE_LOCK = (MIGRATE, 0)
# Roles are cluster-wide, so two runners must not write them at the same moment.
ROLES_LOCK = (MIGRATE, 1)
MAINTENANCE_DATABASE = 'postgres'


def alembic_config() -> Config:
    return Config(str(REPO_ROOT / 'alembic.ini'))


def _conninfo(
    settings: DatabaseSettings, user: str, password: str, dbname: str | None = None
) -> str:
    return psycopg.conninfo.make_conninfo(
        host=settings.direct_host,
        port=settings.port,
        dbname=dbname or settings.name,
        user=user,
        password=password,
        sslmode=settings.sslmode,
        application_name='op-migrate',
    )


def _owner_connection(settings: DatabaseSettings) -> Connection:
    url = URL.create(
        'postgresql+psycopg',
        username=settings.owner_user,
        password=settings.password('owner'),
        host=settings.direct_host,
        port=settings.port,
        database=settings.name,
        query={'sslmode': settings.sslmode, 'application_name': 'op-migrate'},
    )
    return create_engine(url, poolclass=pool.NullPool).connect()


def upgrade(settings: DatabaseSettings, *, super_user: str = SUPER_USER) -> None:
    super_password = settings.password('super')
    # Roles belong to the cluster, and an advisory lock belongs to one database, so they are
    # created from the maintenance database every cluster has.
    with psycopg.connect(
        _conninfo(settings, super_user, super_password, MAINTENANCE_DATABASE), autocommit=True
    ) as bootstrap:
        bootstrap.execute('SELECT pg_advisory_lock(%s, %s)', ROLES_LOCK)
        try:
            roles.ensure_roles(bootstrap, settings)
        finally:
            bootstrap.execute('SELECT pg_advisory_unlock(%s, %s)', ROLES_LOCK)
    with psycopg.connect(
        _conninfo(settings, super_user, super_password), autocommit=True
    ) as target:
        roles.prepare_database(target, settings)

    with _owner_connection(settings) as conn:
        conn.execute(
            text('SELECT pg_advisory_lock(:a, :b)'), {'a': MIGRATE_LOCK[0], 'b': MIGRATE_LOCK[1]}
        )
        conn.commit()
        try:
            cfg = alembic_config()
            cfg.attributes['connection'] = conn
            command.upgrade(cfg, 'head')
            conn.commit()
            with psycopg.connect(
                _conninfo(settings, settings.owner_user, settings.password('owner')),
                autocommit=True,
            ) as grants:
                roles.apply_grants(grants, settings)
        finally:
            conn.rollback()
            conn.execute(
                text('SELECT pg_advisory_unlock(:a, :b)'),
                {'a': MIGRATE_LOCK[0], 'b': MIGRATE_LOCK[1]},
            )
            conn.commit()


def sync_embedding_catalog() -> None:
    """Insert the catalog rows a project create needs (it fails closed without them)."""
    with (
        one_off_connection('app', application_name='op-migrate-catalog') as conn,
        Session(bind=conn) as session,
    ):
        sync_catalog(session)


def post_upgrade() -> None:
    """After the schema is at head: the embedding catalog, then the OpenSearch migration."""
    sync_embedding_catalog()
    asyncio.run(run_with_engine(opensearch_run(), 'op-migrate', one_shot=True))


def current(settings: DatabaseSettings) -> str | None:
    """The revision the database is at, or ``None`` before the first migration."""
    with _owner_connection(settings) as conn:
        return MigrationContext.configure(conn).get_current_revision()


def check(settings: DatabaseSettings) -> None:
    """Raise when the models and the migrated database differ (``alembic check``)."""
    with _owner_connection(settings) as conn:
        cfg = alembic_config()
        cfg.attributes['connection'] = conn
        command.check(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('upgrade', 'current', 'check'))
    args = parser.parse_args(argv)
    settings = DatabaseSettings.from_env()
    if args.action == 'upgrade':
        upgrade(settings)
        post_upgrade()
    elif args.action == 'current':
        sys.stdout.write(f'{current(settings)}\n')
    else:
        check(settings)
    return 0


if __name__ == '__main__':
    sys.exit(main())
