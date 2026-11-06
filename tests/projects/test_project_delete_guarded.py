"""RN-1: the immutability triggers let a delete through only while the project job holds the flag."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from db._research_seed import make_calibration, make_profile, ready_partition
from psycopg import errors
from sqlalchemy.exc import IntegrityError

from projects._delete_rows import bound_engine, connect, count, deleting_project
from src.services.projects.delete import purge_project_rows


if TYPE_CHECKING:
    import uuid

    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

IMMUTABLE = errors.IntegrityConstraintViolation


def _seed(conn, project) -> dict[str, uuid.UUID]:
    doc, _ = make_profile(conn, project)
    conn.execute(
        'INSERT INTO config_revisions (config_id, revision, body, body_sha256, created_by) '
        "VALUES (%s, 2, '{}', 'y', 'test')",
        (doc,),
    )
    conn.execute('UPDATE config_documents SET current_revision = 2 WHERE id = %s', (doc,))
    return {
        'doc': doc,
        'calibration': make_calibration(conn, project),
        'partition': ready_partition(conn, project),
    }


def test_a_project_with_ready_data_deletes_with_the_flag(pg: PgDatabase) -> None:
    with connect(pg) as conn:
        project = deleting_project(conn)
        ids = _seed(conn, project)
        with bound_engine(pg):
            totals = asyncio.run(purge_project_rows(project))
        assert totals['config_revisions'] == 2
        assert totals['cluster_calibrations'] == 1
        assert totals['cluster_partitions'] == 1
        assert count(conn, 'config_documents', 'id', ids['doc']) == 0
        assert count(conn, 'cluster_partitions', 'id', ids['partition']) == 0


def test_the_same_deletes_without_the_flag_raise(pg: PgDatabase) -> None:
    """A deleting project is not enough: the setting must name this project in this transaction."""
    with connect(pg) as conn:
        project = deleting_project(conn)
        ids = _seed(conn, project)
        for statement, key in (
            ('DELETE FROM config_revisions WHERE config_id = %s', 'doc'),
            ('DELETE FROM cluster_calibrations WHERE id = %s', 'calibration'),
            ('DELETE FROM cluster_partitions WHERE id = %s', 'partition'),
        ):
            with pytest.raises(IMMUTABLE):
                conn.execute(statement, (ids[key],))


def test_the_flag_of_another_project_or_an_active_project_does_not_open_the_guard(
    pg: PgDatabase,
) -> None:
    with connect(pg) as conn:
        project = deleting_project(conn)
        other = deleting_project(conn)
        ids = _seed(conn, project)
        with conn.transaction():
            conn.execute("SELECT set_config('op.project_delete', %s, true)", (str(other),))
            with pytest.raises(IMMUTABLE):
                conn.execute(
                    'DELETE FROM cluster_calibrations WHERE id = %s', (ids['calibration'],)
                )
        conn.execute(
            "UPDATE projects SET status = 'active', pre_delete_status = NULL WHERE id = %s",
            (project,),
        )
        with conn.transaction():
            conn.execute("SELECT set_config('op.project_delete', %s, true)", (str(project),))
            with pytest.raises(IMMUTABLE):
                conn.execute(
                    'DELETE FROM cluster_calibrations WHERE id = %s', (ids['calibration'],)
                )


def test_an_active_project_is_never_purged(pg: PgDatabase) -> None:
    with connect(pg) as conn:
        project = deleting_project(conn)
        _seed(conn, project)
        conn.execute(
            "UPDATE projects SET status = 'active', pre_delete_status = NULL WHERE id = %s",
            (project,),
        )
        with bound_engine(pg), pytest.raises(IntegrityError) as raised:
            asyncio.run(purge_project_rows(project))
        assert 'immutable' in str(raised.value)
