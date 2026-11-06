"""A child row gets its project from its partition, so no scope loses sight of it (review RR-18)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import add_cluster, bind_scope, make_identity, make_partition, make_project
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture(scope='module')
def world(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        a = make_project(conn, 'child-a')
        b = make_project(conn, 'child-b')
        return {
            'a': a,
            'b': b,
            'part_a': make_partition(conn, a),
            'part_b': make_partition(conn, b),
            'core': make_partition(conn, None, target='face'),
            'uid_a': make_identity(conn, a),
            'uid_core': make_identity(conn, None, 'face'),
        }


def test_a_membership_with_a_null_project_gets_the_partitions_project(pg, world) -> None:
    with psycopg.connect(pg.conninfo('app')) as conn:
        bind_scope(conn, 'project', world['a'])
        conn.execute(
            'INSERT INTO cluster_memberships (partition_id, project_id, item_key, state) '
            "VALUES (%s, NULL, 'k', 'unassigned')",
            (world['part_a'],),
        )
        assert conn.execute(
            'SELECT project_id FROM cluster_memberships WHERE partition_id = %s', (world['part_a'],)
        ).fetchone() == (world['a'],)


def test_a_partition_cluster_with_a_null_project_gets_the_partitions_project(pg, world) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        add_cluster(conn, world['part_a'], world['uid_a'])
        assert conn.execute(
            'SELECT project_id FROM partition_clusters WHERE partition_id = %s', (world['part_a'],)
        ).fetchone() == (world['a'],)


def test_a_given_project_cannot_override_the_partitions(pg, world) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute(
            'INSERT INTO cluster_memberships (partition_id, project_id, item_key, state) '
            "VALUES (%s, %s, 'forged', 'unassigned')",
            (world['part_b'], world['a']),
        )
        assert conn.execute(
            "SELECT project_id FROM cluster_memberships WHERE item_key = 'forged'"
        ).fetchone() == (world['b'],)


def test_a_project_scope_cannot_write_into_another_projects_partition(pg, world) -> None:
    with psycopg.connect(pg.conninfo('app')) as conn:
        bind_scope(conn, 'project', world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            conn.execute(
                'INSERT INTO cluster_memberships (partition_id, item_key, state) '
                "VALUES (%s, 'x', 'unassigned')",
                (world['part_b'],),
            )


def test_a_core_partition_keeps_its_null_project_and_stays_in_core_scope(pg, world) -> None:
    with psycopg.connect(pg.conninfo('app')) as conn:
        bind_scope(conn, 'core')
        conn.execute(
            'INSERT INTO cluster_memberships (partition_id, item_key, state) '
            "VALUES (%s, 'face-1', 'unassigned')",
            (world['core'],),
        )
        assert conn.execute(
            "SELECT project_id FROM cluster_memberships WHERE item_key = 'face-1'"
        ).fetchone() == (None,)
        conn.commit()
        bind_scope(conn, 'project', world['a'])
        assert conn.execute(
            "SELECT count(*) FROM cluster_memberships WHERE item_key = 'face-1'"
        ).fetchone() == (0,)
        conn.rollback()
        bind_scope(conn, 'core')
        assert conn.execute('SELECT count(*) FROM cluster_partitions').fetchone()[0] == 1


def test_a_membership_of_a_missing_partition_is_refused(pg, world) -> None:
    """The single-column key catches what the composite key skips for a NULL project (X-1)."""
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        for project in (None, world['a']):
            with pytest.raises(errors.ForeignKeyViolation):
                conn.execute(
                    'INSERT INTO cluster_memberships (partition_id, project_id, item_key, state) '
                    "VALUES (%s, %s, 'k', 'unassigned')",
                    (uuid.uuid4(), project),
                )
