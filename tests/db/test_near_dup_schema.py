"""Revision 0003_near_duplicates: keys, the project copy, one current run, scope isolation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import bind_scope, make_profile, make_project
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

_RUN = (
    'INSERT INTO near_dup_runs (project_id, level, mode, status, params, space, algorithm_version, '
    'profile_config_id, profile_revision, finished_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, '
    "CASE WHEN %s = 'building' THEN NULL ELSE now() END) RETURNING id"
)
_PROFILES: dict[uuid.UUID, tuple[uuid.UUID, int]] = {}
_SHA = 'a' * 64
_HASH = 'b' * 32


def _run(conn, project, level='image', status='current'):
    profile = (None, None)
    if project is not None:
        if project not in _PROFILES:
            _PROFILES[project] = make_profile(conn, project, 'near_dup_profile')
        profile = _PROFILES[project]
    return conn.execute(
        _RUN,
        (project, level, 'full', status, '{}', 'pe@abc:pe_embedding', 'nd-1', *profile, status),
    ).fetchone()[0]


@pytest.fixture(scope='module')
def world(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return {'a': make_project(conn, 'nd-a'), 'b': make_project(conn, 'nd-b')}


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


def test_two_current_runs_of_one_project_and_level_are_refused(conn, world) -> None:
    _run(conn, world['a'], 'item')
    with pytest.raises(errors.UniqueViolation, match='uq_near_dup_runs_current'):
        _run(conn, world['a'], 'item')
    _run(conn, world['a'], 'item', 'superseded')
    _run(conn, world['b'], 'item')


def test_only_one_building_run_per_project_and_level(conn, world) -> None:
    _run(conn, world['b'], 'image', 'building')
    with pytest.raises(errors.UniqueViolation, match='uq_near_dup_runs_building'):
        _run(conn, world['b'], 'image', 'building')


def test_the_core_level_has_no_project_and_one_current_run(conn) -> None:
    with pytest.raises(errors.CheckViolation):
        _run(conn, None, 'image')
    _run(conn, None, 'core')
    with pytest.raises(errors.UniqueViolation, match='uq_near_dup_runs_current'):
        _run(conn, None, 'core')


def test_groups_and_members_take_their_project_from_the_run(conn, world) -> None:
    run = _run(conn, world['a'], 'image', 'superseded')
    group = conn.execute(
        'INSERT INTO near_dup_groups (run_id, project_id, purpose, lineage_id, representative_id, '
        "member_count, min_edge_cosine, min_cosine_to_rep) VALUES (%s, %s, 'dedup', %s, 'x', 2, 1, 1) "
        'RETURNING id, project_id',
        (run, world['b'], uuid.uuid4()),
    ).fetchone()
    assert group[1] == world['a']
    member_project = conn.execute(
        'INSERT INTO near_dup_members (run_id, purpose, member_id, group_id, is_representative, '
        "cosine_to_rep) VALUES (%s, 'dedup', 'x', %s, true, 1) RETURNING project_id",
        (run, group[0]),
    ).fetchone()[0]
    assert member_project == world['a']


def test_one_representative_per_group(conn, world) -> None:
    run = _run(conn, world['a'], 'image', 'superseded')
    group = conn.execute(
        'INSERT INTO near_dup_groups (run_id, purpose, lineage_id, representative_id, '
        "member_count, min_edge_cosine, min_cosine_to_rep) VALUES (%s, 'leak', %s, 'x', 2, 1, 1) "
        'RETURNING id',
        (run, uuid.uuid4()),
    ).fetchone()[0]
    sql = (
        'INSERT INTO near_dup_members (run_id, purpose, member_id, group_id, is_representative, '
        "cosine_to_rep) VALUES (%s, 'leak', %s, %s, true, 1)"
    )
    conn.execute(sql, (run, 'm1', group))
    with pytest.raises(errors.UniqueViolation, match='uq_near_dup_members_rep'):
        conn.execute(sql, (run, 'm2', group))


def test_an_imohash_may_repeat_and_a_sha256_is_stored_once_per_project(conn, world) -> None:
    sql = (
        'INSERT INTO image_contents (project_id, imohash, content_sha256, image_id, image_path, '
        'size_bytes) VALUES (%s, %s, %s, %s, %s, 1) ON CONFLICT DO NOTHING RETURNING id'
    )
    for index, owner in enumerate((world['a'], world['b'], None)):
        assert conn.execute(sql, (owner, _HASH, None, f'img-{index}', '/p')).fetchone() is not None
        assert conn.execute(sql, (owner, _HASH, None, f'dup-{index}', '/p')).fetchone() is not None
        assert conn.execute(sql, (owner, _HASH, _SHA, f'sha-{index}', '/p')).fetchone() is not None
        assert conn.execute(sql, (owner, _HASH, _SHA, f'again-{index}', '/p')).fetchone() is None
    assert conn.execute(sql, (world['a'], _HASH, None, 'img-0', '/p')).fetchone() is None
    with pytest.raises(errors.CheckViolation):
        conn.execute(sql, (world['a'], 'not-a-hash', None, 'bad', '/p'))


def test_a_pair_is_stored_in_order(conn, world) -> None:
    with pytest.raises(errors.CheckViolation):
        conn.execute(
            'INSERT INTO near_dup_edges (project_id, level, space, a_id, b_id, kind, score) '
            "VALUES (%s, 'image', 's', 'b', 'a', 'embedding', 0.99)",
            (world['a'],),
        )
    with pytest.raises(errors.CheckViolation):
        conn.execute(
            'INSERT INTO near_dup_decisions (project_id, level, a_id, b_id, decision, decided_by) '
            "VALUES (%s, 'image', 'b', 'a', 'same', 'u')",
            (world['a'],),
        )


def test_a_pending_member_is_one_row_per_project_level_and_member(conn, world) -> None:
    sql = (
        'INSERT INTO near_dup_pending (project_id, level, member_id, reason) '
        "VALUES (%s, 'image', 'm', %s)"
    )
    conn.execute(sql, (world['a'], 'added'))
    with pytest.raises(errors.UniqueViolation):
        conn.execute(sql, (world['a'], 'deleted'))
    conn.execute(sql, (None, 'added'))
    with pytest.raises(errors.UniqueViolation):
        conn.execute(sql, (None, 'added'))


def _visible(pg, table: str, kind: str, project=None) -> int:
    with psycopg.connect(pg.conninfo('app')) as conn:
        bind_scope(conn, kind, project)
        return conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]  # fixed names


def test_a_project_scope_sees_only_its_rows_in_every_table(pg, conn, world) -> None:
    _run(conn, None, 'core', 'superseded')
    for project in (world['a'], world['b']):
        run = _run(conn, project, 'image', 'superseded')
        group = conn.execute(
            'INSERT INTO near_dup_groups (run_id, purpose, lineage_id, representative_id, '
            "member_count, min_edge_cosine, min_cosine_to_rep) VALUES (%s, 'dedup', %s, 'x', 2, 1, 1) "
            'RETURNING id',
            (run, uuid.uuid4()),
        ).fetchone()[0]
        conn.execute(
            'INSERT INTO near_dup_members (run_id, purpose, member_id, group_id, is_representative, '
            "cosine_to_rep) VALUES (%s, 'dedup', 'x', %s, true, 1)",
            (run, group),
        )
    for owner in (world['a'], world['b']):  # no row may come from another test of the module
        conn.execute(
            'INSERT INTO image_contents (project_id, imohash, image_id, image_path, size_bytes) '
            "VALUES (%s, %s, 'scope-img', '/p', 1) ON CONFLICT DO NOTHING",
            (owner, 'c' * 32),
        )
        conn.execute(
            'INSERT INTO near_dup_pending (project_id, level, member_id, reason) '
            "VALUES (%s, 'image', 'scope-m', 'added') ON CONFLICT DO NOTHING",
            (owner,),
        )
    conn.execute(
        'INSERT INTO near_dup_decisions (project_id, level, a_id, b_id, decision, decided_by) '
        "VALUES (%s, 'item', 'a', 'b', 'same', 'u')",
        (world['a'],),
    )
    for table in ('image_contents', 'near_dup_runs', 'near_dup_groups', 'near_dup_members',
                  'near_dup_pending', 'near_dup_decisions'):  # fmt: skip
        total = _visible(pg, table, 'deployment')
        mine = _visible(pg, table, 'project', world['a'])
        theirs = _visible(pg, table, 'project', world['b'])
        core = _visible(pg, table, 'core')
        assert mine + theirs <= total, table
        assert core <= total, table
        if table != 'near_dup_decisions':
            assert mine < total, table
    assert _visible(pg, 'near_dup_decisions', 'project', world['b']) == 0


def test_a_project_scope_cannot_write_another_projects_run(pg, world) -> None:
    with psycopg.connect(pg.conninfo('app')) as c:
        bind_scope(c, 'project', world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            _run(c, world['b'], 'item', 'superseded')
