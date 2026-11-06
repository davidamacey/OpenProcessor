"""A ready partition does not change; retention, job pruning and project delete still work."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import add_membership, make_job, make_partition, make_project, ready_partition
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

IMMUTABLE = errors.IntegrityConstraintViolation


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


@pytest.fixture
def project(conn):
    return make_project(conn, f'p{uuid.uuid4().hex[:12]}')


def _set_deleting(conn, project) -> None:
    conn.execute(
        "UPDATE projects SET status = 'deleting', pre_delete_status = 'active' WHERE id = %s",
        (project,),
    )
    conn.execute("SELECT set_config('op.project_delete', %s, false)", (str(project),))


def test_updating_the_memberships_or_clusters_of_a_ready_partition_raises(conn, project) -> None:
    partition = ready_partition(conn, project)
    with pytest.raises(IMMUTABLE, match='cluster_partition_immutable'):
        conn.execute(
            "UPDATE cluster_memberships SET score = 0.5 WHERE partition_id = %s AND item_key = 'b'",
            (partition,),
        )
    with pytest.raises(IMMUTABLE, match='cluster_partition_immutable'):
        conn.execute('UPDATE partition_clusters SET size = 9 WHERE partition_id = %s', (partition,))


def test_a_building_partition_still_changes(conn, project) -> None:
    partition = make_partition(conn, project)
    add_membership(conn, partition, 'a')
    assert (
        conn.execute(
            'UPDATE cluster_memberships SET score = 0.5 WHERE partition_id = %s', (partition,)
        ).rowcount
        == 1
    )


def test_a_ready_partition_changes_only_its_status(conn, project) -> None:
    partition = ready_partition(conn, project)
    with pytest.raises(IMMUTABLE, match='cluster_partition_immutable'):
        conn.execute('UPDATE cluster_partitions SET seed = 2 WHERE id = %s', (partition,))
    conn.execute("UPDATE cluster_partitions SET status = 'discarded' WHERE id = %s", (partition,))
    assert conn.execute(
        'SELECT status FROM cluster_partitions WHERE id = %s', (partition,)
    ).fetchone() == ('discarded',)


def test_pruning_the_job_of_a_ready_partition_passes_and_clears_the_reference(
    conn, project
) -> None:
    job = make_job(conn, project, 'cluster_run')
    parent = ready_partition(conn, project, job=job)
    child = ready_partition(conn, project, parent=parent)
    conn.execute('DELETE FROM jobs WHERE id = %s', (job,))
    assert conn.execute(
        'SELECT job_id FROM cluster_partitions WHERE id = %s', (parent,)
    ).fetchone() == (None,)
    # but a job cannot be swapped for another one
    other = make_job(conn, project)
    with pytest.raises(IMMUTABLE):
        conn.execute('UPDATE cluster_partitions SET job_id = %s WHERE id = %s', (other, child))


def test_retiring_a_superseded_ready_parent_clears_the_child_reference(conn, project) -> None:
    parent = ready_partition(conn, project)
    child = ready_partition(conn, project, parent=parent)
    with pytest.raises(IMMUTABLE, match='discard it first'):
        conn.execute('DELETE FROM cluster_partitions WHERE id = %s', (parent,))
    # retention: discard first, delete the children rows in batches, then the partition
    conn.execute("UPDATE cluster_partitions SET status = 'discarded' WHERE id = %s", (parent,))
    conn.execute('DELETE FROM cluster_memberships WHERE partition_id = %s', (parent,))
    conn.execute('DELETE FROM partition_clusters WHERE partition_id = %s', (parent,))
    conn.execute('DELETE FROM cluster_partitions WHERE id = %s', (parent,))
    assert conn.execute(
        'SELECT parent_partition_id, status FROM cluster_partitions WHERE id = %s', (child,)
    ).fetchone() == (None, 'ready')


def test_a_cascade_delete_of_a_discarded_partition_passes(conn, project) -> None:
    partition = ready_partition(conn, project)
    conn.execute("UPDATE cluster_partitions SET status = 'discarded' WHERE id = %s", (partition,))
    conn.execute('DELETE FROM cluster_partitions WHERE id = %s', (partition,))
    for table in ('cluster_memberships', 'partition_clusters'):
        assert conn.execute(  # nosec B608  fixed table names
            f'SELECT count(*) FROM {table} WHERE partition_id = %s', (partition,)
        ).fetchone() == (0,)


def test_a_child_delete_under_a_ready_partition_needs_the_project_delete_setting(
    conn, project
) -> None:
    partition = ready_partition(conn, project)
    for table in ('cluster_memberships', 'partition_clusters'):
        with pytest.raises(IMMUTABLE, match='cluster_partition_immutable'):
            conn.execute(f'DELETE FROM {table} WHERE partition_id = %s', (partition,))  # nosec B608
    _set_deleting(conn, project)
    conn.execute('DELETE FROM cluster_memberships WHERE partition_id = %s', (partition,))
    conn.execute('DELETE FROM partition_clusters WHERE partition_id = %s', (partition,))
    conn.execute('DELETE FROM cluster_partitions WHERE id = %s', (partition,))


def test_the_setting_alone_does_not_open_another_project_or_an_active_one(conn, project) -> None:
    partition = ready_partition(conn, project)
    other = make_project(conn, f'p{uuid.uuid4().hex[:12]}')
    conn.execute("SELECT set_config('op.project_delete', %s, false)", (str(project),))
    with pytest.raises(IMMUTABLE):  # the project is not marked deleting
        conn.execute('DELETE FROM cluster_memberships WHERE partition_id = %s', (partition,))
    _set_deleting(conn, other)
    with pytest.raises(IMMUTABLE):  # the flag names another project
        conn.execute('DELETE FROM cluster_memberships WHERE partition_id = %s', (partition,))


def test_deleting_a_project_with_ready_partitions_cascades(conn, project) -> None:
    parent = ready_partition(conn, project, job=make_job(conn, project))
    ready_partition(conn, project, parent=parent)
    conn.execute('DELETE FROM projects WHERE id = %s', (project,))
    assert conn.execute(
        'SELECT count(*) FROM cluster_partitions WHERE project_id = %s', (project,)
    ).fetchone() == (0,)
    assert conn.execute(
        'SELECT count(*) FROM cluster_memberships WHERE project_id = %s', (project,)
    ).fetchone() == (0,)


def test_a_core_ready_partition_is_never_deleted_directly(conn) -> None:
    partition = ready_partition(conn, None, target='face')
    with pytest.raises(IMMUTABLE, match='discard it first'):
        conn.execute('DELETE FROM cluster_partitions WHERE id = %s', (partition,))
