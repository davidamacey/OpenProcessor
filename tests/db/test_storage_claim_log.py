"""The claim log is append only, and only a guarded job deletes (15a sections 2.2 and 8). Pending the wave run.

The guard is the pattern of ``op.project_delete`` (02a 1.2): a transaction-local setting that names a
running ``erasure_jobs`` row of the project. Controls: no setting, a setting that names a job of another
project, a finished job, and text that is not an id each leave the log refusing the delete.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import bind_scope
from _storage_seed import analyser, scalar, seed_world
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture(scope='module')
def world(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return {'a': seed_world(conn, 'log-a'), 'b': seed_world(conn, 'log-b')}


def _app(pg: PgDatabase, project: uuid.UUID):
    conn = psycopg.connect(pg.conninfo('app'))
    bind_scope(conn, 'project', project)
    return conn


def test_an_update_of_a_claim_is_refused(pg, world) -> None:
    with (
        _app(pg, world['a']['project']) as conn,
        pytest.raises(errors.IntegrityConstraintViolation, match='claims_append_only'),
    ):
        conn.execute("UPDATE claims SET actor = 'x' WHERE claim_id = %s", (world['a']['claim'],))


def test_a_delete_of_a_claim_is_refused(pg, world) -> None:
    with (
        _app(pg, world['a']['project']) as conn,
        pytest.raises(errors.IntegrityConstraintViolation, match='claims_append_only'),
    ):
        conn.execute('DELETE FROM claims WHERE claim_id = %s', (world['a']['claim'],))


def test_even_the_superuser_cannot_update_a_claim(pg, world) -> None:
    with (
        psycopg.connect(pg.conninfo('super')) as conn,
        pytest.raises(errors.IntegrityConstraintViolation),
    ):
        conn.execute("UPDATE claims SET actor = 'x' WHERE claim_id = %s", (world['a']['claim'],))


def test_a_claim_is_withdrawn_once(pg, world) -> None:
    with (
        _app(pg, world['a']['project']) as conn,
        pytest.raises(errors.UniqueViolation, match='uq_claims_withdraws'),
    ):
        conn.execute(
            "INSERT INTO claims (project_id, actor_kind, actor, kind, subject_kind, withdraws_claim_id) VALUES (%s, 'human', 'api', 'withdraw', 'claim', %s)",
            (world['a']['project'], world['a']['claim']),
        )


def test_a_running_erasure_job_of_the_project_lets_the_delete_through(pg) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as su:
        w = seed_world(su, 'log-erase')
    with _app(pg, w['project']) as conn:
        conn.execute("SELECT set_config('op.erasure_job', %s, true)", (str(w['erasure']),))
        deleted = conn.execute('DELETE FROM claims WHERE project_id = %s', (w['project'],))
        assert deleted.rowcount == 2
        conn.commit()


@pytest.mark.parametrize('case', ['none', 'other_project', 'finished', 'garbage'])
def test_a_delete_without_a_valid_guard_is_refused(pg, world, case: str) -> None:
    mine, other = world['a'], world['b']
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as su:
        finished = scalar(
            su,
            "INSERT INTO erasure_jobs (project_id, subject_kind, state, finished_at) VALUES (%s, 'photo', 'done', now()) RETURNING id",
            (mine['project'],),
        )
    setting = {
        'none': '',
        'other_project': str(other['erasure']),
        'finished': str(finished),
        'garbage': 'not-a-uuid',
    }[case]
    with _app(pg, mine['project']) as conn:
        conn.execute("SELECT set_config('op.erasure_job', %s, true)", (setting,))
        with pytest.raises(errors.IntegrityConstraintViolation, match='claims_append_only'):
            conn.execute('DELETE FROM claims WHERE claim_id = %s', (mine['claim'],))


def test_deleting_a_photo_is_refused_while_it_would_take_claims_with_it(pg, world) -> None:
    mine = world['a']
    with (
        _app(pg, mine['project']) as conn,
        pytest.raises(errors.IntegrityConstraintViolation, match='claims_append_only'),
    ):
        conn.execute('DELETE FROM photos WHERE photo_key = %s', (mine['photo'],))


def test_erasing_a_photo_removes_its_rows_its_claims_and_their_withdraws(pg) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as su:
        w = seed_world(su, 'log-photo')
    with _app(pg, w['project']) as conn:
        conn.execute("SELECT set_config('op.erasure_job', %s, true)", (str(w['erasure']),))
        conn.execute('DELETE FROM photos WHERE photo_key = %s', (w['photo'],))
        conn.commit()
    with psycopg.connect(pg.conninfo('super')) as su:
        for table in (
            'claims',
            'items',
            'detections',
            'item_state',
            'box_state',
            'region_boxes',
            'box_texts',
            'faces',
            'face_state',
        ):
            assert su.execute(
                f'SELECT count(*) FROM {table} WHERE project_id = %s', (w['project'],)
            ).fetchone() == (0,), table  # nosec B608


def test_the_project_delete_job_may_delete_claims_under_its_own_guard(pg) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as su:
        w = seed_world(su, 'log-delete')
        su.execute(
            "UPDATE projects SET status = 'deleting', pre_delete_status = 'active' WHERE id = %s",
            (w['project'],),
        )
    with _app(pg, w['project']) as conn:
        conn.execute("SELECT set_config('op.project_delete', %s, true)", (str(w['project']),))
        assert (
            conn.execute('DELETE FROM claims WHERE project_id = %s', (w['project'],)).rowcount == 2
        )
        conn.commit()


def test_deleting_the_project_row_takes_its_claims_with_it(pg) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as su:
        w = seed_world(su, 'log-cascade')
        su.execute('DELETE FROM projects WHERE id = %s', (w['project'],))
        assert su.execute(
            'SELECT count(*) FROM claims WHERE project_id = %s', (w['project'],)
        ).fetchone() == (0,)


def test_an_analyser_is_never_changed_or_deleted(pg, world) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as su:
        found = analyser(su)
    for statement in (
        "UPDATE analysers SET version = '2' WHERE analyser_id = %s",
        'DELETE FROM analysers WHERE analyser_id = %s',
    ):
        with (
            psycopg.connect(pg.conninfo('super')) as su,
            pytest.raises(errors.IntegrityConstraintViolation, match='analyser_immutable'),
        ):
            su.execute(statement, (found,))


def test_the_claim_checks_refuse_a_claim_that_could_not_be_found_again(pg, world) -> None:
    mine = world['a']
    bad = [
        # an item claim needs the photo, the wire id, the box the person saw and the decode frame
        "INSERT INTO claims (project_id, actor_kind, actor, kind, subject_kind) VALUES (%(p)s, 'human', 'api', 'class_set', 'item')",
        # a withdraw names its target, and only a withdraw does
        "INSERT INTO claims (project_id, actor_kind, actor, kind, subject_kind) VALUES (%(p)s, 'human', 'api', 'withdraw', 'claim')",
        # a kind is paired with its own subject kind
        "INSERT INTO claims (project_id, actor_kind, actor, kind, subject_kind) VALUES (%(p)s, 'human', 'api', 'class_set', 'photo')",
        # a box of the wrong length
        "INSERT INTO claims (project_id, actor_kind, actor, kind, subject_kind, subject_bbox) VALUES (%(p)s, 'human', 'api', 'duplicate_verdict', 'pair', '{0.1}')",
    ]
    for statement in bad:
        with _app(pg, mine['project']) as conn, pytest.raises(errors.IntegrityError):
            conn.execute(statement, {'p': mine['project']})
