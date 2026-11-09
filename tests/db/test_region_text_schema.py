"""Revision 0005_region_text: tables, one open pass per configuration, one group per box, no text."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import make_job, make_profile, make_project
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

TABLES = ('region_text_runs', 'region_text_consensus_groups', 'region_text_consensus_members')
# The only text-valued columns: codes, version labels, ids and the issuing jurisdiction. The
# readable plate text lives in OpenSearch (D-79) and must never get a column here.
TEXT_COLUMNS = {
    'region_text_runs': {'trigger', 'reader', 'reader_version', 'created_by'},
    'region_text_consensus_groups': {'basis', 'method', 'tier', 'jurisdiction'},
    'region_text_consensus_members': {'item_id', 'box_id'},
}
_RUN = (
    'INSERT INTO region_text_runs (project_id, trigger, reader, reader_version, '
    "profile_config_id, profile_revision, created_by) VALUES (%s, %s, 'rec', 'v1', %s, %s, 'test') "
    'RETURNING id'
)
_GROUP = (
    'INSERT INTO region_text_consensus_groups (project_id, run_id, basis, method, n_frames, '
    'n_agree, tier, jurisdiction, jurisdiction_conf) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) '
    'RETURNING id'
)
_MEMBER = (
    'INSERT INTO region_text_consensus_members (group_id, project_id, item_id, box_id, weight, '
    'agrees) VALUES (%s, %s, %s, %s, 1, true)'
)


@pytest.fixture(scope='module')
def world(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        a = make_project(conn, 'rt-a')
        b = make_project(conn, 'rt-b')
        return {'a': a, 'b': b, 'profile': make_profile(conn, a, 'region_profile')}


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


def _run(conn, project, trigger='region_pass', profile=(None, None)):
    return conn.execute(_RUN, (project, trigger, *profile)).fetchone()[0]


def _group(conn, project, run, *, n_frames=3, n_agree=2, jurisdiction=None, conf=None):
    return conn.execute(
        _GROUP,
        (
            project,
            run,
            'near_dup',
            'best_frame',
            n_frames,
            n_agree,
            'confident',
            jurisdiction,
            conf,
        ),
    ).fetchone()[0]


def test_every_table_index_and_policy_is_present(conn) -> None:
    tables = {
        r[0]
        for r in conn.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'region_text%'")
    }
    assert tables == set(TABLES)
    indexes = {
        r[0]
        for r in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename LIKE 'region_text%'"
        )
    }
    assert {
        'ix_region_text_runs_project', 'uq_region_text_runs_open_pass',
        'ix_region_text_consensus_groups_run', 'uq_region_text_consensus_members_box',
    } <= indexes  # fmt: skip
    policies = {
        r[0]
        for r in conn.execute(
            "SELECT policyname FROM pg_policies WHERE tablename LIKE 'region_text%'"
        )
    }
    assert policies == {f'p_{t}_project' for t in TABLES}
    forced = conn.execute(
        "SELECT count(*) FROM pg_class WHERE relname LIKE 'region_text%' AND relkind = 'r' "
        'AND relrowsecurity AND relforcerowsecurity'
    ).fetchone()[0]
    assert forced == len(TABLES)


def test_the_tables_hold_no_text_column_besides_the_listed_codes(conn) -> None:
    rows = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public' "
        "AND table_name LIKE 'region_text%' AND data_type IN ('text', 'character varying', 'character')"
    ).fetchall()
    found: dict[str, set[str]] = {}
    for table, column in rows:
        found.setdefault(table, set()).add(column)
    assert found == TEXT_COLUMNS


def test_the_consensus_marker_is_an_allowed_project_state_key(conn, world) -> None:
    conn.execute(
        "INSERT INTO project_state (project_id, key, value) VALUES (%s, 'region_text_consensus_marker', '{}')",
        (world['a'],),
    )
    with pytest.raises(errors.CheckViolation, match='ck_project_state_key'):
        conn.execute(
            "INSERT INTO project_state (project_id, key, value) VALUES (%s, 'other_marker', '{}')",
            (world['a'],),
        )


def test_a_second_open_pass_for_one_configuration_is_refused(conn, world) -> None:
    profile = world['profile']
    first = _run(conn, world['a'], profile=profile)
    with pytest.raises(errors.UniqueViolation, match='uq_region_text_runs_open_pass'):
        _run(conn, world['a'], profile=profile)
    conn.execute('UPDATE region_text_runs SET finished_at = now() WHERE id = %s', (first,))
    _run(conn, world['a'], profile=profile)
    _run(conn, world['a'], 'reread', profile=profile)
    _run(conn, world['a'], 'reread', profile=profile)


def test_a_configuration_without_a_profile_is_one_configuration(conn, world) -> None:
    _run(conn, world['b'])
    with pytest.raises(errors.UniqueViolation, match='uq_region_text_runs_open_pass'):
        _run(conn, world['b'])


def test_a_profile_pair_and_trigger_are_checked(conn, world) -> None:
    with pytest.raises(errors.CheckViolation, match='ck_region_text_runs_trigger'):
        _run(conn, world['a'], 'rekey')
    with pytest.raises(errors.CheckViolation, match='ck_region_text_runs_profile_pair'):
        _run(conn, world['a'], 'reread', (world['profile'][0], None))


def test_a_run_refuses_a_profile_revision_that_does_not_exist(conn, world) -> None:
    with pytest.raises(errors.ForeignKeyViolation, match='fk_region_text_runs_profile'):
        _run(conn, world['a'], 'reread', (world['profile'][0], 99))


def test_a_deleted_job_leaves_its_run(conn, world) -> None:
    job = make_job(conn, world['a'])
    run = conn.execute(
        'INSERT INTO region_text_runs (project_id, job_id, trigger, reader, reader_version, created_by) '
        "VALUES (%s, %s, 'consensus', 'rec', 'v1', 'test') RETURNING id",
        (world['a'], job),
    ).fetchone()[0]
    conn.execute('DELETE FROM jobs WHERE id = %s', (job,))
    assert conn.execute('SELECT job_id FROM region_text_runs WHERE id = %s', (run,)).fetchone() == (
        None,
    )


def test_a_box_in_two_current_groups_is_a_unique_violation(conn, world) -> None:
    run = _run(conn, world['a'], 'consensus')
    first, second = _group(conn, world['a'], run), _group(conn, world['a'], run)
    conn.execute(_MEMBER, (first, world['a'], 'item-1', 'box-1'))
    conn.execute(_MEMBER, (first, world['a'], 'item-1', 'box-2'))
    with pytest.raises(errors.UniqueViolation, match='uq_region_text_consensus_members_box'):
        conn.execute(_MEMBER, (second, world['a'], 'item-1', 'box-1'))


def test_the_same_box_ids_in_another_project_are_a_different_box(conn, world) -> None:
    run = _run(conn, world['b'], 'consensus')
    group = _group(conn, world['b'], run)
    conn.execute(_MEMBER, (group, world['b'], 'item-1', 'box-1'))


def test_a_group_and_a_member_must_belong_to_their_parents_project(conn, world) -> None:
    run = _run(conn, world['a'], 'consensus')
    with pytest.raises(
        errors.ForeignKeyViolation, match='fk_region_text_consensus_groups_run_project'
    ):
        _group(conn, world['b'], run)
    group = _group(conn, world['a'], run)
    with pytest.raises(
        errors.ForeignKeyViolation, match='fk_region_text_consensus_members_group_project'
    ):
        conn.execute(_MEMBER, (group, world['b'], 'item-9', 'box-9'))


@pytest.mark.parametrize(
    ('kwargs', 'constraint'),
    [
        ({'n_frames': 0, 'n_agree': 0}, 'ck_region_text_consensus_groups_n_frames'),
        ({'n_frames': 2, 'n_agree': 3}, 'ck_region_text_consensus_groups_n_agree'),
        ({'n_agree': -1}, 'ck_region_text_consensus_groups_n_agree'),
        ({'jurisdiction': 'x', 'conf': 1.5}, 'ck_region_text_consensus_groups_jurisdiction_conf'),
    ],
)
def test_group_counts_and_jurisdiction_confidence_are_bounded(
    conn, world, kwargs, constraint
) -> None:
    run = _run(conn, world['a'], 'consensus')
    with pytest.raises(errors.CheckViolation, match=constraint):
        _group(conn, world['a'], run, **kwargs)


def test_the_jurisdiction_and_its_confidence_are_nullable(conn, world) -> None:
    run = _run(conn, world['a'], 'consensus')
    plain = _group(conn, world['a'], run)
    read = _group(conn, world['a'], run, jurisdiction='synthetic-state', conf=0.75)
    rows = {
        r[0]: r[1:]
        for r in conn.execute(
            'SELECT id, jurisdiction, jurisdiction_conf FROM region_text_consensus_groups '
            'WHERE id = ANY(%s)',
            ([plain, read],),
        ).fetchall()
    }
    assert rows[plain] == (None, None)
    assert rows[read] == ('synthetic-state', 0.75)


def test_deleting_a_run_deletes_its_groups_and_members(conn, world) -> None:
    run = _run(conn, world['a'], 'consensus')
    group = _group(conn, world['a'], run)
    conn.execute(_MEMBER, (group, world['a'], 'gone', 'gone'))
    conn.execute('DELETE FROM region_text_runs WHERE id = %s', (run,))
    assert conn.execute(
        "SELECT count(*) FROM region_text_consensus_members WHERE item_id = 'gone'"
    ).fetchone() == (0,)


def test_a_project_sees_only_its_own_rows(pg, world) -> None:
    with psycopg.connect(pg.conninfo('app')) as app:
        app.execute("SELECT set_config('op.scope', 'project', true)")
        app.execute("SELECT set_config('op.project_id', %s, true)", (str(world['b']),))
        seen = {r[0] for r in app.execute('SELECT DISTINCT project_id FROM region_text_runs')}
        assert seen <= {world['b']}
        with pytest.raises(errors.InsufficientPrivilege):
            app.execute(_RUN, (world['a'], 'reread', None, None))


def test_a_uuid_parent_that_does_not_exist_is_refused(conn, world) -> None:
    with pytest.raises(errors.ForeignKeyViolation):
        _group(conn, world['a'], uuid.uuid4())
