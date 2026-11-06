"""The migrated database holds the objects of 02d_research_seam_schema.md section 2.

The expectations are in ``_research_plan.py``, written by hand from the plan. The exact
comparisons (no extra table, constraint, trigger or policy) live in ``test_schema_matches_plan.py``,
which merges those sets with 02a's; this file adds what only 02d states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import _research_plan as plan
import psycopg
import pytest


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture(scope='module')
def catalog(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        yield conn


def _set(conn, query: str, *params) -> set:
    return {tuple(r) if len(r) > 1 else r[0] for r in conn.execute(query, params).fetchall()}


def test_every_planned_table_exists(catalog) -> None:
    found = _set(catalog, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    assert found >= plan.TABLES


def test_constraints_indexes_and_names(catalog) -> None:
    constraints = _set(catalog, 'SELECT conname FROM pg_constraint')
    assert constraints >= plan.UNIQUE_CONSTRAINTS
    assert constraints >= plan.NAMED_CHECKS
    indexes = _set(catalog, 'SELECT indexname FROM pg_indexes')
    assert indexes >= plan.PARTIAL_INDEXES | plan.PLAIN_INDEXES
    nulls = _set(
        catalog,
        'SELECT i.relname FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid '
        'WHERE x.indnullsnotdistinct',
    )
    assert nulls >= plan.NULLS_NOT_DISTINCT


def test_the_one_primary_space_index_is_partial_on_role(catalog) -> None:
    (definition,) = catalog.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'uq_embedding_spaces_one_primary'"
    ).fetchone()
    assert 'UNIQUE' in definition
    assert "role = 'primary'" in definition


def test_foreign_keys_and_their_delete_rules(catalog) -> None:
    found = _set(
        catalog,
        'SELECT c.conrelid::regclass::text, a.attname, c.confrelid::regclass::text, c.confdeltype::text '
        'FROM pg_constraint c JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1] '
        "WHERE c.contype = 'f'",
    )
    assert found >= plan.FOREIGN_KEYS


@pytest.mark.parametrize(
    ('table', 'columns', 'target'),
    [
        ('cluster_memberships', ['partition_id', 'project_id'], 'cluster_partitions'),
        ('partition_clusters', ['partition_id', 'project_id'], 'cluster_partitions'),
        ('cluster_partitions', ['profile_config_id', 'profile_revision'], 'config_revisions'),
        ('assist_loop_runs', ['preset_config_id', 'preset_revision'], 'config_revisions'),
        ('cluster_identities', ['project_id', 'promoted_class_id'], 'project_classes'),
    ],
)
def test_composite_references_name_both_columns(catalog, table, columns, target) -> None:
    rows = catalog.execute(
        'SELECT (SELECT array_agg(a.attname ORDER BY k.ord) '
        'FROM unnest(c.conkey) WITH ORDINALITY k(attnum, ord) '
        'JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum) '
        "FROM pg_constraint c WHERE c.contype = 'f' "
        'AND c.conrelid = %s::regclass AND c.confrelid = %s::regclass',
        (table, target),
    ).fetchall()
    assert columns in [r[0] for r in rows]


def test_triggers_functions_and_policies(catalog) -> None:
    triggers = _set(
        catalog, 'SELECT tgrelid::regclass::text, tgname FROM pg_trigger WHERE NOT tgisinternal'
    )
    assert triggers >= plan.TRIGGERS
    functions = _set(
        catalog, "SELECT proname FROM pg_proc WHERE pronamespace = 'public'::regnamespace"
    )
    assert functions >= plan.FUNCTIONS
    assert _set(catalog, 'SELECT tablename, policyname FROM pg_policies') >= plan.POLICIES


def test_the_triggers_fire_before_the_events_the_plan_names(catalog) -> None:
    rows = dict(
        catalog.execute(
            "SELECT tgrelid::regclass::text || '.' || tgname, tgtype FROM pg_trigger "
            'WHERE NOT tgisinternal'
        ).fetchall()
    )
    before, insert, delete, update = 2, 4, 8, 16
    for key, events in {
        'cluster_partitions.cluster_partitions_ready_immutable': update | delete,
        'cluster_memberships.cluster_children_ready_immutable': update | delete,
        'cluster_memberships.cluster_children_project_from_parent': insert,
        'embedding_model_versions.embedding_model_versions_immutable': update,
    }.items():
        assert rows[key] & before, key
        assert rows[key] & (insert | delete | update) == events, key


def test_the_triggers_that_read_a_protected_table_are_security_definer(catalog) -> None:
    definer = _set(
        catalog,
        "SELECT proname FROM pg_proc WHERE prosecdef AND proname LIKE 'cluster\\_%%'",
    )
    assert definer == {
        'cluster_calibrations_immutable',
        'cluster_partitions_ready_immutable',
        'cluster_children_ready_immutable',
        'cluster_children_project_from_parent',
    }


def test_row_level_security_is_forced_where_the_plan_says(catalog) -> None:
    protected = _set(
        catalog,
        "SELECT relname FROM pg_class WHERE relnamespace = 'public'::regnamespace "
        "AND relkind = 'r' AND relrowsecurity AND relforcerowsecurity AND relname = ANY(%s)",
        sorted(plan.TABLES),
    )
    assert protected == plan.TABLES - plan.UNPROTECTED


def test_the_default_calibration_policy_is_select_only(catalog) -> None:
    (command,) = catalog.execute(
        "SELECT cmd FROM pg_policies WHERE policyname = 'p_cluster_calibrations_default_read'"
    ).fetchone()
    assert command == 'SELECT'


def test_entity_keys_default_to_uuidv7(catalog) -> None:
    keyed = _set(
        catalog,
        "SELECT table_name FROM information_schema.columns WHERE table_schema = 'public' "
        "AND column_name = 'id' AND data_type = 'uuid' AND column_default = 'uuidv7()'",
    )
    assert {
        'cluster_calibrations', 'cluster_partitions', 'cluster_activations', 'assist_loop_runs',
        'embedding_model_versions', 'embedding_spaces', 'embedding_reindex_runs',
        'embedding_evaluation_runs', 'cluster_backend_validations', 'assist_loop_reviews',
    } <= keyed  # fmt: skip


def test_every_table_of_a_research_revision_has_a_stated_writer_and_reader(catalog) -> None:
    from src.db.table_use import TABLE_USE

    created = plan.TABLES  # revision 0002; 0003 and 0004 add their own entries with their tables
    assert created - set(TABLE_USE) == set(), 'tables without a TABLE_USE entry'
    assert set(TABLE_USE) - created == set() or set(TABLE_USE) <= _all_tables(catalog)
    assert all(len(use) == 3 and all(use) for use in TABLE_USE.values())


def _all_tables(catalog) -> set:
    return _set(catalog, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
