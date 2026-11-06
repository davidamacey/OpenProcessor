"""Seeding and an engine for the project-delete row tests: rows go in as the superuser."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import psycopg
from db._research_seed import add_cluster, make_identity, make_job, make_partition, make_project
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from src.db.engine import bind_async_engine


if TYPE_CHECKING:
    from collections.abc import Iterator

    from fixtures.postgres import PgDatabase


def connect(pg: PgDatabase) -> psycopg.Connection[Any]:
    return psycopg.connect(pg.conninfo('super'), autocommit=True)


def deleting_project(conn: Any, slug: str | None = None) -> uuid.UUID:
    """A project already marked ``deleting``, the only state the guarded deletes accept."""
    project = make_project(conn, slug or f'p{uuid.uuid4().hex[:10]}')
    conn.execute(
        "UPDATE projects SET status = 'deleting', pre_delete_status = 'active' WHERE id = %s",
        (project,),
    )
    return project


def big_partition(conn: Any, project: uuid.UUID, rows: int) -> uuid.UUID:
    """A ``ready`` partition with ``rows`` memberships (written while it was still building)."""
    partition = make_partition(conn, project)
    uid = make_identity(conn, project)
    add_cluster(conn, partition, uid)
    conn.execute(
        'INSERT INTO cluster_memberships (partition_id, item_key, state) '
        "SELECT %s, 'k' || g, 'unassigned' FROM generate_series(1, %s) g",
        (partition, rows),
    )
    conn.execute("UPDATE cluster_partitions SET status = 'ready' WHERE id = %s", (partition,))
    return partition


def job_with_items(conn: Any, project: uuid.UUID, rows: int) -> uuid.UUID:
    job = make_job(conn, project, 'reprocess')
    conn.execute(
        "INSERT INTO job_items (job_id, position, item_id) SELECT %s, g, 'i' || g "
        'FROM generate_series(1, %s) g',
        (job, rows),
    )
    return job


def count(conn: Any, table: str, column: str, value: Any) -> int:
    return int(
        conn.execute(f'SELECT count(*) FROM {table} WHERE {column} = %s', (value,)).fetchone()[0]
    )


@contextmanager
def bound_engine(pg: PgDatabase, *, statement_timeout: str = '30s') -> Iterator[None]:
    """An async engine for the app role with a short statement timeout, bound for the block."""
    engine = create_async_engine(
        pg.url('app'),
        poolclass=pool.NullPool,
        connect_args={'options': f'-c statement_timeout={statement_timeout}'},
    )
    with bind_async_engine(engine):
        yield
