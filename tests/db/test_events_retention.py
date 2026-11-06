"""delete_older_than removes only rows past the age, in bounded batches."""

from __future__ import annotations

from datetime import timedelta

import psycopg
import pytest

from src.db.repos.events import delete_older_than
from src.db.session import sync_deployment_tx, sync_session_factory


pytestmark = pytest.mark.db


@pytest.fixture(scope='module')
def pg(pg_events):
    """The cursor waits for the cluster's oldest open transaction: use a server of its own."""
    return pg_events


def _count(pg) -> int:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return conn.execute('SELECT count(*) FROM events').fetchone()[0]


def test_only_rows_older_than_the_age_are_deleted_one_batch_at_a_time(pg_env) -> None:
    with psycopg.connect(pg_env.conninfo('super'), autocommit=True) as conn:
        conn.execute('DELETE FROM events')
        conn.execute(
            "INSERT INTO events (type, payload, created_at) SELECT 'old', '{}', now() - interval '25 hours' "
            'FROM generate_series(1, 5)'
        )
        conn.execute(
            "INSERT INTO events (type, payload, created_at) SELECT 'fresh', '{}', now() - interval '23 hours' "
            'FROM generate_series(1, 3)'
        )
    with sync_session_factory()() as session, sync_deployment_tx(session, 'retention test'):
        deleted = [delete_older_than(session, timedelta(hours=24), batch=2) for _ in range(3)]
    assert deleted == [2, 2, 1]
    with psycopg.connect(pg_env.conninfo('super'), autocommit=True) as conn:
        left = [r[0] for r in conn.execute('SELECT type FROM events')]
    assert left == ['fresh'] * 3
