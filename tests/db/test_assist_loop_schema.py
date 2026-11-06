"""The auto-label loop record: checks, cascades, row-level security through ``run_id``."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import bind_scope, make_job, make_loop_run, make_profile, make_project
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

_NODE = (
    'INSERT INTO assist_loop_nodes (run_id, path, iteration, level, member_count, eligible_count, '
    "membership_digest, snapshot_storage_key, step, decision) VALUES (%s, %s, 1, 0, 10, 10, 'd', "
    "'k', %s, %s)"
)
_REVIEW = (
    'INSERT INTO assist_loop_reviews (run_id, item_id, queue, machine_source, shown_class_name, '
    "decision, human_class_name, served_at, reviewer) VALUES (%s, %s, %s, 'direct', %s, %s, %s, "
    "now(), 'test')"
)


@pytest.fixture(scope='module')
def world(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        a, b = make_project(conn, 'loop-a'), make_project(conn, 'loop-b')
        runs = {}
        for name, project in (('a', a), ('b', b)):
            run = make_loop_run(conn, project)
            conn.execute(
                "INSERT INTO assist_loop_iterations (run_id, ordinal, status) VALUES (%s, 1, 'open')",
                (run,),
            )
            conn.execute(_NODE, (run, 'r0', 'open', None))
            conn.execute(
                'INSERT INTO assist_loop_budgets (run_id, pool, unit, limit_value) '
                "VALUES (%s, 'vlm', 'crops', 100)",
                (run,),
            )
            runs[name] = run
        return {'a': a, 'b': b, 'run_a': runs['a'], 'run_b': runs['b']}


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


def test_a_project_sees_the_loop_children_of_its_own_runs_only(pg, world) -> None:
    for table, expected in (
        ('assist_loop_runs', 1),
        ('assist_loop_iterations', 1),
        ('assist_loop_nodes', 1),
        ('assist_loop_budgets', 1),
    ):
        with psycopg.connect(pg.conninfo('app')) as c:
            bind_scope(c, 'project', world['a'])
            seen = c.execute(f'SELECT count(*) FROM {table}').fetchone()[0]  # nosec B608  fixed names
            assert seen == expected, table


def test_a_project_cannot_write_a_child_of_another_projects_run(pg, world) -> None:
    with psycopg.connect(pg.conninfo('app')) as c:
        bind_scope(c, 'project', world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            c.execute(_NODE, (world['run_b'], 'r1', 'open', None))


def test_a_project_cannot_reach_another_projects_run_row(pg, world) -> None:
    with psycopg.connect(pg.conninfo('app')) as c:
        bind_scope(c, 'project', world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            c.execute(
                'INSERT INTO assist_loop_runs (project_id, job_id, request, strategy, seed, pinned, '
                "created_by) VALUES (%s, %s, '{}', 'cluster_sampled', 1, '{}', 'x')",
                (world['b'], uuid.uuid4()),
            )


def test_the_budget_refuses_an_over_reserve(conn, world) -> None:
    with pytest.raises(errors.CheckViolation, match='no_over_reserve'):
        conn.execute(
            'UPDATE assist_loop_budgets SET reserved = 101 WHERE run_id = %s', (world['run_a'],)
        )
    conn.execute(
        'UPDATE assist_loop_budgets SET reserved = 60, spent = 40 WHERE run_id = %s',
        (world['run_a'],),
    )
    with pytest.raises(errors.CheckViolation, match='no_over_reserve'):
        conn.execute(
            'UPDATE assist_loop_budgets SET spent = 41 WHERE run_id = %s', (world['run_a'],)
        )
    conn.execute(
        'UPDATE assist_loop_budgets SET reserved = 0, spent = 0 WHERE run_id = %s',
        (world['run_a'],),
    )


def test_prompt_tokens_have_no_over_reserve_check(conn, world) -> None:
    conn.execute(
        'INSERT INTO assist_loop_budgets (run_id, pool, unit, limit_value, spent) '
        "VALUES (%s, 'vlm', 'prompt_tokens', 10, 50)",
        (world['run_a'],),
    )


def test_the_ledger_follows_its_budget_row(conn, world) -> None:
    ledger = (
        'INSERT INTO assist_loop_budget_ledger (run_id, request_id, pool, unit, entry, amount, '
        "fencing_token) VALUES (%s, %s, %s, %s, 'reserve', 1, 3)"
    )
    with pytest.raises(errors.ForeignKeyViolation):
        conn.execute(ledger, (world['run_a'], uuid.uuid4(), 'audit', 'crops'))
    conn.execute(ledger, (world['run_a'], uuid.uuid4(), 'vlm', 'crops'))


def test_the_blind_audit_row_refuses_a_shown_class(conn, world) -> None:
    with pytest.raises(errors.CheckViolation, match='blind_audit_hides_class'):
        conn.execute(_REVIEW, (world['run_a'], 'i1', 'blind_audit', 'car', 'accept', 'car'))
    conn.execute(_REVIEW, (world['run_a'], 'i1', 'blind_audit', None, 'accept', 'car'))
    conn.execute(_REVIEW, (world['run_a'], 'i1', 'verify', 'car', 'accept', 'car'))
    with pytest.raises(errors.UniqueViolation):
        conn.execute(_REVIEW, (world['run_a'], 'i1', 'verify', 'car', 'accept', 'car'))
    with pytest.raises(errors.CheckViolation, match='human_class_iff_accepted'):
        conn.execute(_REVIEW, (world['run_a'], 'i2', 'verify', 'car', 'reject', 'car'))


def test_node_rules(conn, world) -> None:
    run = world['run_a']
    with pytest.raises(errors.CheckViolation, match='path'):
        conn.execute(_NODE, (run, 'x1', 'open', None))
    with pytest.raises(errors.CheckViolation, match='done_iff_decided'):
        conn.execute(_NODE, (run, 'r1', 'done', None))
    with pytest.raises(errors.CheckViolation, match='done_iff_decided'):
        conn.execute(_NODE, (run, 'r1', 'open', 'pass'))
    conn.execute(_NODE, (run, 'r1.2', 'done', 'pass'))
    with pytest.raises(errors.ForeignKeyViolation):  # an iteration that does not exist
        conn.execute(_NODE.replace('1, 0, 10', '9, 0, 10'), (run, 'r2', 'open', None))


def test_a_proposal_that_answered_names_a_class(conn, world) -> None:
    insert = (
        'INSERT INTO assist_loop_proposals (run_id, item_id, role, proposer_kind, proposer_ref, '
        'request_id, outcome, class_name) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'
    )
    run = world['run_a']
    with pytest.raises(errors.CheckViolation, match='answered_has_class'):
        conn.execute(insert, (run, 'i1', 'primary', 'vlm', 'c:1', uuid.uuid4(), 'answered', None))
    conn.execute(insert, (run, 'i1', 'primary', 'vlm', 'c:1', uuid.uuid4(), 'answered', 'car'))
    conn.execute(insert, (run, 'i2', 'primary', 'vlm', 'c:1', uuid.uuid4(), 'failed', None))


def test_a_run_needs_both_preset_columns_or_neither(conn) -> None:
    project = make_project(conn, 'loop-preset')
    doc, revision = make_profile(conn, project, 'assist_loop_preset')
    insert = (
        'INSERT INTO assist_loop_runs (project_id, job_id, preset_config_id, preset_revision, '
        "request, strategy, seed, pinned, created_by) VALUES (%s, %s, %s, %s, '{}', "
        "'cluster_sampled', 1, '{}', 'test')"
    )
    with pytest.raises(errors.CheckViolation, match='preset_pair'):
        conn.execute(insert, (project, make_job(conn, project), doc, None))
    with pytest.raises(errors.ForeignKeyViolation):
        conn.execute(insert, (project, make_job(conn, project), doc, revision + 5))
    conn.execute(insert, (project, make_job(conn, project), doc, revision))
    # the profile revision a run pinned cannot be deleted under it
    with pytest.raises(errors.ForeignKeyViolation):
        conn.execute('DELETE FROM config_documents WHERE id = %s', (doc,))


def test_deleting_a_run_removes_its_children_and_a_job_with_a_run_is_kept(conn) -> None:
    project = make_project(conn, 'loop-delete')
    job = make_job(conn, project, 'assist_loop')
    run = make_loop_run(conn, project, job)
    conn.execute(
        "INSERT INTO assist_loop_iterations (run_id, ordinal, status) VALUES (%s, 1, 'open')",
        (run,),
    )
    conn.execute(_NODE, (run, 'r0', 'open', None))
    with pytest.raises(errors.ForeignKeyViolation):  # jobs keep-list: NO ACTION
        conn.execute('DELETE FROM jobs WHERE id = %s', (job,))
    conn.execute('DELETE FROM assist_loop_runs WHERE id = %s', (run,))
    for table in ('assist_loop_iterations', 'assist_loop_nodes'):
        assert conn.execute(  # nosec B608  fixed names
            f'SELECT count(*) FROM {table} WHERE run_id = %s', (run,)
        ).fetchone() == (0,)


def test_vlm_usage_is_hourly_and_per_project(pg, world, conn) -> None:
    endpoint = conn.execute(
        'INSERT INTO config_documents (project_id, kind, name, current_revision, updated_by) '
        "VALUES (NULL, 'vlm_endpoint', 'local', 1, 'test') RETURNING id"
    ).fetchone()[0]
    insert = (
        'INSERT INTO vlm_usage_hourly (project_id, endpoint_config_id, source, hour) '
        "VALUES (%s, %s, 'assist_loop', %s)"
    )
    with pytest.raises(errors.CheckViolation, match='hour'):
        conn.execute(insert, (world['a'], endpoint, '2026-11-05 10:30:00+00'))
    conn.execute(insert, (world['a'], endpoint, '2026-11-05 10:00:00+00'))
    with psycopg.connect(pg.conninfo('app')) as c:
        bind_scope(c, 'project', world['b'])
        assert c.execute('SELECT count(*) FROM vlm_usage_hourly').fetchone() == (0,)
