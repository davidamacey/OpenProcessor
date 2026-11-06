"""The research seams of revision 0001: the core scope, guarded project deletes and the new checks."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _rls_seed import ids, seed_project
from psycopg import errors

from src.db import session as db_session
from src.db.session import ScopeNotAllowedError, core_tx, sync_session_factory
from src.domain import job_states as js


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

_JOB = (
    'INSERT INTO jobs (id, project_id, kind, queue, status, max_queue_wait, requested_by) '
    "VALUES (%s, %s, 'noop', 'jobs', 'queued', interval '1 hour', 'test')"
)


def _scope(conn, kind: str, project: uuid.UUID | None = None) -> None:
    conn.execute("SELECT set_config('op.scope', %s, true)", (kind,))
    conn.execute("SELECT set_config('op.project_id', %s, true)", (str(project or ''),))


@pytest.fixture(scope='module')
def world(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        a = seed_project(conn, 'seam-a')
        core_job = uuid.uuid4()
        conn.execute(_JOB, (core_job, None))
        conn.execute(
            "INSERT INTO events (project_id, type, payload) VALUES (NULL, 'job.state', '{}')"
        )
        conn.execute(
            'INSERT INTO config_documents (id, project_id, kind, name, current_revision, updated_by) '
            "VALUES (%s, NULL, 'vlm_endpoint', 'endpoint', 1, 'test')",
            (uuid.uuid4(),),
        )
    return {'a': a, 'core_job': core_job}


def _app(pg):
    return psycopg.connect(pg.conninfo('app'))


@pytest.mark.parametrize('table', ['jobs', 'events', 'config_documents'])
def test_a_project_scope_cannot_see_core_rows(pg, world, table: str) -> None:
    with _app(pg) as conn:
        _scope(conn, 'project', world['a'])
        nulls = conn.execute(f'SELECT count(*) FROM {table} WHERE project_id IS NULL').fetchone()[0]
    assert nulls == 0


@pytest.mark.parametrize('table', ['jobs', 'events', 'config_documents'])
def test_the_core_scope_sees_core_rows_and_no_project_rows(pg, world, table: str) -> None:
    with _app(pg) as conn:
        _scope(conn, 'core')
        rows = conn.execute(f'SELECT project_id FROM {table}').fetchall()
    assert rows
    assert {r[0] for r in rows} == {None}


@pytest.mark.parametrize('table', ['jobs', 'events', 'config_documents'])
def test_the_deployment_scope_sees_both(pg, world, table: str) -> None:
    with _app(pg) as conn:
        _scope(conn, 'deployment')
        projects = {r[0] for r in conn.execute(f'SELECT project_id FROM {table}').fetchall()}
    assert {None, world['a']} <= projects


def test_a_core_job_writes_its_attempt_and_item_rows_under_the_core_scope(pg, world) -> None:
    with _app(pg) as conn:
        _scope(conn, 'core')
        conn.execute(
            "INSERT INTO job_attempts (job_id, attempt, fencing_token, owner) VALUES (%s, 1, 1, 'w')",
            (world['core_job'],),
        )
        conn.execute(
            "INSERT INTO job_items (job_id, position, item_id) VALUES (%s, 0, 'i')",
            (world['core_job'],),
        )
        assert conn.execute('SELECT count(*) FROM job_attempts').fetchone()[0] == 1
        conn.rollback()


def test_a_project_scope_cannot_write_a_core_jobs_children(pg, world) -> None:
    with _app(pg) as conn:
        _scope(conn, 'project', world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            conn.execute(
                'INSERT INTO job_attempts (job_id, attempt, fencing_token, owner) '
                "VALUES (%s, 1, 1, 'w')",
                (world['core_job'],),
            )


def test_the_core_scope_cannot_write_a_project_row(pg, world) -> None:
    with _app(pg) as conn:
        _scope(conn, 'core')
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            conn.execute(_JOB, (uuid.uuid4(), world['a']))


def test_a_core_job_can_be_created_under_the_core_scope_only(pg, world) -> None:
    with _app(pg) as conn:
        _scope(conn, 'core')
        conn.execute(_JOB, (uuid.uuid4(), None))
        conn.rollback()
        _scope(conn, 'project', world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            conn.execute(_JOB, (uuid.uuid4(), None))


def test_core_tx_outside_its_allowlist_raises() -> None:
    code = compile('result = core_tx(session, "probe")', '/srv/app/src/services/rogue.py', 'exec')
    namespace = {'__name__': 'src.services.rogue', 'core_tx': core_tx, 'session': object()}
    with pytest.raises(ScopeNotAllowedError, match=r'src\.services\.rogue'):
        exec(code, namespace)  # a caller frame with a chosen module name


def test_core_tx_from_a_listed_module_opens_the_core_scope(pg_env) -> None:
    code = compile('result = core_tx(session, "probe")', '/srv/app/src/routers/persons.py', 'exec')
    with sync_session_factory()() as session:
        namespace = {
            '__name__': 'src.routers.persons',
            'core_tx': db_session.sync_core_tx,
            'session': session,
        }
        exec(code, namespace)
        with namespace['result']:
            from sqlalchemy import text

            row = session.execute(
                text(
                    "SELECT current_setting('op.scope', true), current_setting('op.project_id', true)"
                )
            ).one()
    assert tuple(row) == ('core', '')


def test_a_paused_job_needs_a_reason(pg, world) -> None:
    with (
        psycopg.connect(pg.conninfo('super'), autocommit=True) as conn,
        pytest.raises(errors.CheckViolation, match='paused_has_reason'),
    ):
        conn.execute(
            'INSERT INTO jobs (kind, queue, status, max_queue_wait, requested_by) '
            "VALUES ('noop', 'jobs', 'paused', interval '1 hour', 'test')"
        )


_DOC = 'INSERT INTO config_documents (project_id, kind, name, current_revision, updated_by) '
_ACTIVATION = (
    'INSERT INTO config_activations (project_id, axis, kind, config_id, revision, activated_by) '
)
_TASK = 'INSERT INTO search_projection_tasks (project_id, kind, payload) '
_REFUSED = [
    (_DOC + "VALUES (%(p)s, 'serving_profile', 'x', 1, 't')", 'ck_config_documents_scope'),
    (_DOC + "VALUES (NULL, 'cluster_profile', 'x', 1, 't')", 'ck_config_documents_scope'),
    (
        _ACTIVATION
        + "SELECT %(p)s, 'clustering_item', 'region_profile', id, 1, 't' FROM config_documents LIMIT 1",
        'ck_config_activations_axis_kind',
    ),
    (
        "INSERT INTO gpu_reservations (gpu_uuid, holder_job_id) VALUES ('GPU-unknown', %(job)s)",
        'fk_gpu_reservations_gpu_uuid_gpu_devices',
    ),
    (
        "INSERT INTO deployment_settings (key, value, updated_by) VALUES ('other', '{}', 't')",
        'ck_deployment_settings_key',
    ),
    (_TASK + "VALUES (%(p)s, 'not_a_kind', '{}')", 'ck_search_projection_tasks_kind'),
    (
        "INSERT INTO project_state (project_id, key, value) VALUES (%(p)s, 'last_clustering', '{}')",
        'ck_project_state_key',
    ),
]


@pytest.mark.parametrize(('statement', 'constraint'), _REFUSED)
def test_the_seam_constraints_refuse_what_the_plan_forbids(
    pg, world, statement: str, constraint: str
) -> None:
    with (
        psycopg.connect(pg.conninfo('super'), autocommit=True) as conn,
        pytest.raises((errors.CheckViolation, errors.ForeignKeyViolation), match=constraint),
    ):
        conn.execute(statement, {'p': world['a'], 'job': ids(world['a'], 'a')['job']})


def test_an_engine_row_without_an_input_contract_is_refused(pg, world) -> None:
    model = ids(world['a'], 'a')['model']
    with (
        psycopg.connect(pg.conninfo('super'), autocommit=True) as conn,
        pytest.raises(errors.NotNullViolation, match='input_contract'),
    ):
        conn.execute(
            'INSERT INTO model_engines (model_id, triton_model_name, precision, input_size, '
            'output_contract, build_settings, build_stamp, max_batch, trt_version, gpu_arch, status) '
            "VALUES (%s, 'x', 'fp16', 640, '{}', '{}', 's', 8, '10', 'sm_86', 'ready')",
            (model,),
        )


def test_a_job_attempt_may_end_yielded(pg, world) -> None:
    with psycopg.connect(pg.conninfo('super')) as conn:
        conn.execute(
            'INSERT INTO job_attempts (job_id, attempt, fencing_token, owner, outcome) '
            "VALUES (%s, 7, 1, 'w', 'yielded')",
            (world['core_job'],),
        )
        conn.rollback()


@pytest.mark.parametrize(
    ('source', 'target'), [('running', 'queued'), ('queued', 'paused'), ('paused', 'queued')]
)
def test_the_yield_pause_and_resume_pairs_are_listed(source: str, target: str) -> None:
    assert (source, target) in js.TRANSITIONS


def test_a_cancelled_job_cannot_be_queued_again() -> None:
    assert ('cancelled', 'queued') not in js.TRANSITIONS


# --- guarded project delete -------------------------------------------------------------------


@pytest.fixture
def document(pg, world):
    """A project config document with two revisions, written as the superuser."""
    doc = uuid.uuid4()
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute(
            'INSERT INTO config_documents (id, project_id, kind, name, current_revision, updated_by) '
            "VALUES (%s, %s, 'prompt_pack', %s, 2, 't')",
            (doc, world['a'], f'doc-{doc}'),
        )
        for revision in (1, 2):
            conn.execute(
                'INSERT INTO config_revisions (config_id, revision, body, body_sha256, created_by) '
                "VALUES (%s, %s, '{}', 'x', 't')",
                (doc, revision),
            )
    yield doc
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute(
            "UPDATE projects SET status = 'active', pre_delete_status = NULL WHERE id = %s",
            (world['a'],),
        )
        conn.execute('DELETE FROM config_documents WHERE id = %s', (doc,))


def _delete_revisions(pg, world, document, *, flag_for: uuid.UUID | None, status: str) -> int:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        pre = (
            ", pre_delete_status = 'active'"
            if status == 'deleting'
            else ', pre_delete_status = NULL'
        )
        conn.execute(f'UPDATE projects SET status = %s{pre} WHERE id = %s', (status, world['a']))
    with _app(pg) as conn:
        _scope(conn, 'deployment')
        if flag_for is not None:
            conn.execute("SELECT set_config('op.project_delete', %s, true)", (str(flag_for),))
        count = conn.execute(
            'DELETE FROM config_revisions WHERE config_id = %s', (document,)
        ).rowcount
        conn.commit()
    return count


def test_a_revision_cannot_be_deleted_without_the_delete_flag(pg, world, document) -> None:
    with pytest.raises(errors.IntegrityConstraintViolation, match='config_revision_immutable'):
        _delete_revisions(pg, world, document, flag_for=None, status='deleting')


def test_a_revision_cannot_be_deleted_with_the_flag_naming_another_project(
    pg, world, document
) -> None:
    with pytest.raises(errors.IntegrityConstraintViolation, match='config_revision_immutable'):
        _delete_revisions(pg, world, document, flag_for=uuid.uuid4(), status='deleting')


def test_a_revision_cannot_be_deleted_while_the_project_is_not_deleting(
    pg, world, document
) -> None:
    with pytest.raises(errors.IntegrityConstraintViolation, match='config_revision_immutable'):
        _delete_revisions(pg, world, document, flag_for=world['a'], status='active')


def test_a_revision_can_be_deleted_by_the_project_delete_job(pg, world, document) -> None:
    assert _delete_revisions(pg, world, document, flag_for=world['a'], status='deleting') == 2


def test_a_revision_is_never_updated(pg, world, document) -> None:
    with (
        psycopg.connect(pg.conninfo('super'), autocommit=True) as conn,
        pytest.raises(errors.IntegrityConstraintViolation, match='config_revision_immutable'),
    ):
        conn.execute(
            'UPDATE config_revisions SET body = \'{"x": 1}\' WHERE config_id = %s', (document,)
        )


def test_deleting_the_document_still_cascades_to_its_revisions(pg, world, document) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute('DELETE FROM config_documents WHERE id = %s', (document,))
        assert (
            conn.execute(
                'SELECT count(*) FROM config_revisions WHERE config_id = %s', (document,)
            ).fetchone()[0]
            == 0
        )
