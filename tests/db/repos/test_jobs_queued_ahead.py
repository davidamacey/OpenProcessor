"""``queued_ahead`` counts the jobs queued before one on its queue, over every project, and no
job of another queue or state."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest

from src.db.repos import jobs_listing
from src.db.session import async_session_factory, deployment_tx


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = [pytest.mark.db, pytest.mark.asyncio, pytest.mark.usefixtures('bound_async_engine')]

_BASE = dt.datetime(2026, 11, 12, 12, 0, tzinfo=dt.UTC)


def _insert(conn: psycopg.Connection, queue: str, status: str, minute: int) -> uuid.UUID:
    live = status in ('running', 'cancelling')
    row = conn.execute(
        'INSERT INTO jobs (kind, queue, status, max_queue_wait, requested_by, queued_at, owner, '
        "lease_expires_at) VALUES ('noop', %s, %s, interval '1 hour', 'test', %s, "
        "CASE WHEN %s THEN 'w:1' END, CASE WHEN %s THEN now() + interval '1 hour' END) "
        'RETURNING id',
        (queue, status, _BASE + dt.timedelta(minutes=minute), live, live),
    ).fetchone()
    assert row is not None
    return row[0]


async def test_only_older_queued_jobs_of_the_same_queue_count(
    repo_env: None, pg: PgDatabase
) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        _insert(conn, 'jobs', 'queued', 0)
        _insert(conn, 'interactive', 'queued', 1)
        _insert(conn, 'interactive', 'running', 2)
        second = _insert(conn, 'interactive', 'queued', 3)
        third = _insert(conn, 'interactive', 'queued', 4)

    async def ahead(job_id: uuid.UUID, minute: int) -> int:
        async with async_session_factory()() as session, deployment_tx(session, 'test count'):
            return await session.run_sync(
                lambda sync: jobs_listing.queued_ahead(
                    sync,
                    queue='interactive',
                    queued_at=_BASE + dt.timedelta(minutes=minute),
                    job_id=job_id,
                )
            )

    assert await ahead(second, 3) == 1
    assert await ahead(third, 4) == 2
