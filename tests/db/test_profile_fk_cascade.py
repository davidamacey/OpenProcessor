"""One project delete reaches every table below it, whatever the references between them (review RR-6).

The referencing pairs are enumerated from the catalog: every foreign key between two tables that a
project delete reaches through ``ON DELETE CASCADE`` and whose own rule is not a cascade. Each is
populated (a new pair that nothing populates fails the test instead of slipping past), then both
delete paths must pass. A pair that fails is declared ``DEFERRABLE INITIALLY DEFERRED`` in its
revision, never worked around here.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import (
    add_cluster,
    add_membership,
    bind_scope,
    make_calibration,
    make_job,
    make_model_version,
    make_partition,
    make_profile,
    make_space,
)
from _rls_seed import seed_project
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

_PAIRS_SQL = """
WITH RECURSIVE reach(rel) AS (
  SELECT 'projects'::regclass::oid
  UNION
  SELECT c.conrelid FROM pg_constraint c JOIN reach r ON c.confrelid = r.rel
  WHERE c.contype = 'f' AND c.confdeltype = 'c'
)
SELECT c.conname, c.conrelid::regclass::text, c.confrelid::regclass::text, c.confdeltype::text,
       (SELECT array_agg(a.attname ORDER BY k.ord)
          FROM unnest(c.conkey) WITH ORDINALITY k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum),
       c.condeferrable, c.condeferred
FROM pg_constraint c
WHERE c.contype = 'f' AND c.confdeltype <> 'c'
  AND c.conrelid IN (SELECT rel FROM reach) AND c.confrelid IN (SELECT rel FROM reach)
  AND c.conrelid <> 'projects'::regclass
ORDER BY 2, 1
"""


@pytest.fixture(scope='module')
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


def _populate_class_merge(conn, project: uuid.UUID) -> None:
    conn.execute(
        'INSERT INTO project_classes (project_id, class_id, name, name_key, merged_into, deprecated) '
        "VALUES (%s, 1, 'van', 'van', 0, true)",
        (project,),
    )


def _populate_research_pairs(conn, project: uuid.UUID) -> None:
    """Every non-cascade reference between two tables of revision 0002 (and into 0001)."""
    profile = make_profile(conn, project, 'prompt_pack')
    calibration = make_calibration(conn, project)
    job = make_job(conn, project, 'cluster_run')
    parent = make_partition(conn, project, status='ready', job=job, profile=profile)
    partition = make_partition(
        conn, project, status='ready', parent=parent, calibration=calibration, profile=profile
    )
    class_name = conn.execute(
        'SELECT class_id FROM project_classes WHERE project_id = %s LIMIT 1', (project,)
    ).fetchone()[0]
    uid = conn.execute(
        'INSERT INTO cluster_identities (project_id, target, level, promoted_class_id, '
        "first_partition_id, last_partition_id) VALUES (%s, 'item', 'class', %s, %s, %s) "
        'RETURNING cluster_uid',
        (project, class_name, parent, partition),
    ).fetchone()[0]
    add_cluster(conn, partition, uid)
    add_membership(conn, partition, 'a', uid)
    conn.execute(
        'INSERT INTO cluster_activations (project_id, target, level, embedding_field, '
        "serving_partition_id, pending_partition_id, updated_by) VALUES (%s, 'item', 'class', 'f', "
        "%s, %s, 'test')",
        (project, partition, parent),
    )
    version = make_model_version(conn, f'm{uuid.uuid4().hex[:10]}')
    space = make_space(conn, project, version)
    conn.execute(
        'INSERT INTO embedding_reindex_runs (project_id, space_id, job_id, mode) '
        "VALUES (%s, %s, %s, 'build')",
        (project, space, make_job(conn, project, 'embedding_reindex')),
    )
    conn.execute(
        'UPDATE embedding_spaces SET built_by_job_id = %s WHERE id = %s',
        (make_job(conn, project, 'embedding_reindex'), space),
    )
    run_job = make_job(conn, project, 'assist_loop')
    preset = make_profile(conn, project, 'assist_loop_preset')
    conn.execute(
        'INSERT INTO assist_loop_runs (project_id, job_id, preset_config_id, preset_revision, '
        "request, strategy, seed, pinned, created_by) VALUES (%s, %s, %s, %s, '{}', "
        "'cluster_sampled', 1, '{}', 'test')",
        (project, run_job, *preset),
    )


def _build_project(conn, slug: str) -> uuid.UUID:
    project = seed_project(conn, slug)
    _populate_class_merge(conn, project)
    _populate_research_pairs(conn, project)
    return project


def _unpopulated(conn) -> list[str]:
    """Blocking pairs (NO ACTION, RESTRICT) with no row; a SET NULL pair cannot refuse a delete."""
    missing = []
    for name, table, _ref, rule, columns, _def, _dfd in conn.execute(_PAIRS_SQL).fetchall():
        if rule not in ('a', 'r'):
            continue
        not_null = ' AND '.join(f'{c} IS NOT NULL' for c in columns)
        count = conn.execute(f'SELECT count(*) FROM {table} WHERE {not_null}').fetchone()[0]  # nosec B608  catalog names
        if count == 0:
            missing.append(f'{table}.{",".join(columns)} ({name})')
    return missing


def test_the_pairs_are_found_from_the_catalog_and_include_the_research_tables(conn) -> None:
    pairs = {(row[1], row[2]) for row in conn.execute(_PAIRS_SQL).fetchall()}
    assert ('cluster_partitions', 'config_revisions') in pairs
    assert ('cluster_partitions', 'cluster_calibrations') in pairs
    assert ('assist_loop_runs', 'config_revisions') in pairs
    assert ('embedding_reindex_runs', 'jobs') in pairs


def test_every_pair_is_populated_by_the_seed(conn) -> None:
    _build_project(conn, 'cascade-seed')
    assert _unpopulated(conn) == []


def test_a_direct_delete_of_the_project_passes(conn) -> None:
    project = _build_project(conn, 'cascade-direct')
    conn.execute('DELETE FROM projects WHERE id = %s', (project,))
    assert conn.execute(
        'SELECT (SELECT count(*) FROM cluster_partitions WHERE project_id = %s), '
        '(SELECT count(*) FROM jobs WHERE project_id = %s), '
        '(SELECT count(*) FROM assist_loop_runs WHERE project_id = %s)',
        (project, project, project),
    ).fetchone() == (0, 0, 0)


def test_the_batched_delete_of_the_project_delete_job_passes(pg, conn) -> None:
    project = _build_project(conn, 'cascade-batched')
    conn.execute(
        "UPDATE projects SET status = 'deleting', pre_delete_status = 'active' WHERE id = %s",
        (project,),
    )
    with psycopg.connect(pg.conninfo('app')) as app:
        bind_scope(app, 'project', project)
        app.execute("SELECT set_config('op.project_delete', %s, true)", (str(project),))
        # children first, in batches; the activations that serve a partition go before it
        for table in ('cluster_memberships', 'partition_clusters', 'cluster_activations'):
            app.execute(f'DELETE FROM {table} WHERE project_id = %s', (project,))  # nosec B608
        app.execute('DELETE FROM cluster_partitions WHERE project_id = %s', (project,))
        app.execute('DELETE FROM projects WHERE id = %s', (project,))
        app.commit()
    assert conn.execute('SELECT count(*) FROM projects WHERE id = %s', (project,)).fetchone() == (
        0,
    )


def test_deleting_a_referenced_profile_document_alone_is_refused(conn) -> None:
    project = _build_project(conn, 'cascade-profile')
    doc = conn.execute(
        'SELECT profile_config_id FROM cluster_partitions WHERE project_id = %s LIMIT 1', (project,)
    ).fetchone()[0]
    with pytest.raises(errors.ForeignKeyViolation):
        conn.execute('DELETE FROM config_documents WHERE id = %s', (doc,))
