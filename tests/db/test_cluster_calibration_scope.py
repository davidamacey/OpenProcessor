"""The deployment default calibration is readable from every scope and writable from few (review RR-3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import bind_scope, make_calibration, make_partition, make_project
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

_INSERT = (
    'INSERT INTO cluster_calibrations (project_id, target, level, embedding_field, embedding_model, '
    "preprocess_id, kind, fitted_on, body, dev_metrics, created_by) VALUES (%s, 'item', 'class', "
    "'f', 'm', 'p', 'assign_bands', 'pin:x', '{}', '{}', 'test')"
)


@pytest.fixture(scope='module')
def world(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        a, b = make_project(conn, 'cal-a'), make_project(conn, 'cal-b')
        return {
            'a': a,
            'b': b,
            'default': make_calibration(conn, None),
            'own_a': make_calibration(conn, a),
            'own_b': make_calibration(conn, b),
        }


def _visible(pg, kind: str, project=None) -> set:
    with psycopg.connect(pg.conninfo('app')) as conn:
        bind_scope(conn, kind, project)
        return {r[0] for r in conn.execute('SELECT id FROM cluster_calibrations')}


def test_a_project_scope_reads_the_default_and_its_own_calibration_only(pg, world) -> None:
    assert _visible(pg, 'project', world['a']) == {world['default'], world['own_a']}
    assert _visible(pg, 'project', world['b']) == {world['default'], world['own_b']}


def test_the_core_scope_reads_the_default_but_no_project_row(pg, world) -> None:
    assert _visible(pg, 'core') == {world['default']}


def test_the_deployment_scope_reads_everything(pg, world) -> None:
    assert _visible(pg, 'deployment') == {world['default'], world['own_a'], world['own_b']}


def test_a_project_scope_cannot_write_a_default_calibration(pg, world) -> None:
    with psycopg.connect(pg.conninfo('app')) as conn:
        bind_scope(conn, 'project', world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            conn.execute(_INSERT, (None,))


def test_a_project_scope_cannot_write_another_projects_calibration(pg, world) -> None:
    with psycopg.connect(pg.conninfo('app')) as conn:
        bind_scope(conn, 'project', world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            conn.execute(_INSERT, (world['b'],))
        conn.rollback()
        bind_scope(conn, 'project', world['a'])
        conn.execute(_INSERT, (world['a'],))


def test_the_deployment_and_core_scopes_write_a_default(pg, world) -> None:
    for kind in ('deployment', 'core'):
        with psycopg.connect(pg.conninfo('app')) as conn:
            bind_scope(conn, kind)
            conn.execute(_INSERT, (None,))


def test_a_project_scoped_partition_run_can_pin_the_default_calibration(pg, world) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        partition = make_partition(conn, world['a'], calibration=world['default'])
    with psycopg.connect(pg.conninfo('app')) as conn:
        bind_scope(conn, 'project', world['a'])
        row = conn.execute(
            'SELECT p.id, c.id FROM cluster_partitions p JOIN cluster_calibrations c '
            'ON c.id = p.calibration_id WHERE p.id = %s',
            (partition,),
        ).fetchone()
    assert row == (partition, world['default'])


def test_a_calibration_never_changes(pg, world) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        with pytest.raises(
            errors.IntegrityConstraintViolation, match='cluster_calibration_immutable'
        ):
            conn.execute(
                'UPDATE cluster_calibrations SET body = \'{"x": 1}\' WHERE id = %s',
                (world['own_a'],),
            )
        with pytest.raises(
            errors.IntegrityConstraintViolation, match='cluster_calibration_immutable'
        ):
            conn.execute('DELETE FROM cluster_calibrations WHERE id = %s', (world['default'],))
        with pytest.raises(
            errors.IntegrityConstraintViolation, match='cluster_calibration_immutable'
        ):
            conn.execute('DELETE FROM cluster_calibrations WHERE id = %s', (world['own_b'],))
        conn.execute(
            'DELETE FROM projects WHERE id = %s', (world['b'],)
        )  # a project delete cascades
        assert conn.execute(
            'SELECT count(*) FROM cluster_calibrations WHERE id = %s', (world['own_b'],)
        ).fetchone() == (0,)
