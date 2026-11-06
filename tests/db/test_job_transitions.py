"""The jobs transition trigger allows exactly the pairs of ``src/domain/job_states.py``."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import psycopg
import pytest
from psycopg import errors

from src.domain import job_states as js


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

# Companion columns the table's CHECKs require for a status; the trigger runs before them.
_COMPANIONS = (
    "finished_at = CASE WHEN %(to)s IN ('completed','completed_with_errors','failed','cancelled') "
    'THEN now() ELSE NULL END, '
    "owner = CASE WHEN %(to)s IN ('running','cancelling') THEN 'w:1' ELSE NULL END, "
    "lease_expires_at = CASE WHEN %(to)s IN ('running','cancelling') THEN now() + interval '2 min' "
    'ELSE NULL END, '
    "pause_reason = CASE WHEN %(to)s = 'paused' THEN 'user' ELSE NULL END"
)
_INSERT = (
    'INSERT INTO jobs (kind, queue, status, max_queue_wait, requested_by, finished_at, owner, '
    "lease_expires_at, pause_reason) VALUES ('noop', 'jobs', %(s)s, interval '1 hour', 'test', "
    "CASE WHEN %(s)s IN ('completed','completed_with_errors','failed','cancelled') THEN now() END, "
    "CASE WHEN %(s)s IN ('running','cancelling') THEN 'w:1' END, "
    "CASE WHEN %(s)s IN ('running','cancelling') THEN now() + interval '2 min' END, "
    "CASE WHEN %(s)s = 'paused' THEN 'user' END) RETURNING id"
)

_ALL_PAIRS = [(a, b) for a, b in itertools.product(js.STATUSES, repeat=2) if a != b]


@pytest.fixture(scope='module')
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


def _move(conn, source: str, target: str) -> None:
    conn.execute('BEGIN')
    try:
        job_id = conn.execute(_INSERT, {'s': source}).fetchone()[0]
        conn.execute(
            f'UPDATE jobs SET status = %(to)s, {_COMPANIONS} WHERE id = %(id)s',
            {'to': target, 'id': job_id},
        )
    finally:
        conn.execute('ROLLBACK')


@pytest.mark.parametrize(('source', 'target'), sorted(js.TRANSITIONS))
def test_a_listed_transition_succeeds(conn, source: str, target: str) -> None:
    _move(conn, source, target)


@pytest.mark.parametrize(('source', 'target'), [p for p in _ALL_PAIRS if p not in js.TRANSITIONS])
def test_every_other_transition_is_refused_by_the_trigger(conn, source: str, target: str) -> None:
    with pytest.raises(errors.CheckViolation, match='invalid_job_transition'):
        _move(conn, source, target)


def test_a_write_that_does_not_change_the_status_is_not_checked(conn) -> None:
    conn.execute('BEGIN')
    try:
        job_id = conn.execute(_INSERT, {'s': 'completed'}).fetchone()[0]
        conn.execute(
            "UPDATE jobs SET status = 'completed', error_code = 'x' WHERE id = %s", (job_id,)
        )
    finally:
        conn.execute('ROLLBACK')
