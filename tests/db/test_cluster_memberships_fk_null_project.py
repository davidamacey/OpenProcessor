"""A membership of a NULL-project partition that does not exist is refused (02d section 6, X-1).

The composite key ``(partition_id, project_id)`` is MATCH SIMPLE and skips a NULL project, so the
single-column key on ``partition_id`` is what holds for core rows.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import add_membership, make_partition
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


def test_a_core_membership_of_a_missing_partition_is_refused(conn) -> None:
    with pytest.raises(errors.ForeignKeyViolation, match='cluster_memberships_partition_id'):
        add_membership(conn, uuid.uuid4(), 'face-1')


def test_a_core_membership_of_a_real_core_partition_is_accepted(conn) -> None:
    partition = make_partition(conn, None, target='face')
    add_membership(conn, partition, 'face-1')
    assert conn.execute(
        'SELECT count(*) FROM cluster_memberships WHERE partition_id = %s', (partition,)
    ).fetchone() == (1,)


def test_the_partition_cluster_key_is_checked_the_same_way(conn) -> None:
    with pytest.raises(errors.ForeignKeyViolation):
        conn.execute(
            'INSERT INTO partition_clusters (partition_id, cluster_uid, display_no, transition, size) '
            "VALUES (%s, %s, 1, 'new', 1)",
            (uuid.uuid4(), uuid.uuid4()),
        )
