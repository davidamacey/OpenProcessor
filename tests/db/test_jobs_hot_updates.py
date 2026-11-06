"""Heartbeat, progress and checkpoint writes on ``jobs`` are heap-only-tuple updates.

A normal UPDATE writes a new row version and a new entry in every index; a HOT update
avoids both, but only while no indexed column changes (PostgreSQL storage-hot). The columns
those writes touch are in no index, so almost every one of them must be HOT.
"""

from __future__ import annotations

import psycopg
import pytest


pytestmark = pytest.mark.db


def test_a_thousand_progress_updates_are_almost_all_hot(pg) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        job = conn.execute(
            'INSERT INTO jobs (kind, queue, status, max_queue_wait, requested_by, owner, '
            "lease_expires_at) VALUES ('noop', 'jobs', 'running', interval '1 hour', 'test', "
            "'w:1', now() + interval '2 min') RETURNING id"
        ).fetchone()[0]
        before = conn.execute(
            'SELECT coalesce(n_tup_upd, 0), coalesce(n_tup_hot_upd, 0) FROM pg_stat_user_tables '
            "WHERE relname = 'jobs'"
        ).fetchone()
        for i in range(1000):
            conn.execute(
                "UPDATE jobs SET progress = jsonb_build_object('done', %s::int), "
                'progress_updated_at = now(), heartbeat_at = now(), '
                "lease_expires_at = now() + interval '2 min', "
                "checkpoint = jsonb_build_object('cursor', %s::int) WHERE id = %s",
                (i, i, job),
            )
        conn.execute('SELECT pg_stat_force_next_flush()')
        after = conn.execute(
            "SELECT n_tup_upd, n_tup_hot_upd FROM pg_stat_user_tables WHERE relname = 'jobs'"
        ).fetchone()
    updates, hot = after[0] - before[0], after[1] - before[1]
    assert updates >= 1000
    assert hot >= 0.95 * updates, f'{hot} of {updates} updates were HOT'
