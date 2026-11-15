"""The migrated database equals 15a sections 2, 8 and 9 (pending the wave run: needs PostgreSQL 18).

``test_schema_matches_plan.py`` merges the same hand-written plan into its exact comparisons of tables,
keys, indexes, policies, triggers and foreign keys; this file adds what only the catalog can answer:
column order and identity keys, the foreign-key index rule, the seeded claim kinds, the ``vec`` schema
and the downgrade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import _storage_plan as plan
import psycopg
import pytest
from alembic import command

from scripts.db import migrate
from src.domain.claim_kinds import KINDS


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture(scope='module')
def catalog(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        yield conn


def test_every_table_has_the_planned_columns_in_order(catalog) -> None:
    rows = catalog.execute(
        'SELECT table_name, array_agg(column_name::text ORDER BY ordinal_position) '
        "FROM information_schema.columns WHERE table_schema = 'public' AND table_name = ANY(%s) "
        'GROUP BY table_name',
        (sorted(plan.TABLES),),
    ).fetchall()
    assert {t: tuple(c) for t, c in rows} == plan.COLUMNS


def test_the_high_volume_keys_are_identity_columns_that_always_generate(catalog) -> None:
    found = dict(
        catalog.execute(
            'SELECT table_name, column_name FROM information_schema.columns '
            "WHERE table_schema = 'public' AND identity_generation = 'ALWAYS' AND data_type = 'bigint'"
        ).fetchall()
    )
    for table, column in plan.IDENTITY_KEYS.items():
        assert found.get(table) == column


def test_every_planned_foreign_key_index_exists_and_is_partial_exactly_when_the_column_is_nullable(
    catalog,
) -> None:
    rows = catalog.execute(
        'SELECT i.relname, t.relname, a.attname, x.indpred IS NOT NULL, a.attnotnull '
        'FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_class t ON t.oid = x.indrelid '
        'JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = x.indkey[0] '
        'WHERE i.relname = ANY(%s)',
        (sorted(plan.FOREIGN_KEY_INDEX_NAMES),),
    ).fetchall()
    assert {r[0] for r in rows} == plan.FOREIGN_KEY_INDEX_NAMES
    for name, table, column, partial, not_null in rows:
        assert name == f'ix_{table}_{column}'
        assert partial == (not not_null), name


def test_every_foreign_key_of_a_new_table_is_covered_by_an_index_unless_it_points_at_a_catalog(
    catalog,
) -> None:
    rows = catalog.execute(
        'SELECT c.conrelid::regclass::text, a.attname, c.confrelid::regclass::text '
        'FROM pg_constraint c JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1] '
        "WHERE c.contype = 'f' AND c.conrelid::regclass::text = ANY(%s) AND NOT EXISTS ("
        '  SELECT 1 FROM pg_index x WHERE x.indrelid = c.conrelid AND x.indkey[0] = c.conkey[1] '
        "  AND (x.indpred IS NULL OR pg_get_expr(x.indpred, x.indrelid) = '(' || a.attname || ' IS NOT NULL)'))",
        (sorted(plan.TABLES),),
    ).fetchall()
    uncovered = sorted((t, c) for t, c, ref in rows if ref not in ('analysers', 'claim_kinds'))
    assert uncovered == []


def test_the_claim_kinds_are_the_domain_catalogue(catalog) -> None:
    rows = set(
        catalog.execute(
            'SELECT kind, subject_kind, layer, pins_geometry, asserts_object FROM claim_kinds'
        ).fetchall()
    )
    assert rows == {
        (k.value, i.subject_kind.value, i.layer.value, i.pins_geometry, i.asserts_object)
        for k, i in KINDS.items()
    }


def test_the_existing_tables_gained_their_columns(catalog) -> None:
    for table, column in plan.ADDED_COLUMNS.items():
        found = catalog.execute(
            'SELECT is_nullable FROM information_schema.columns WHERE table_name = %s AND column_name = %s',
            (table, column),
        ).fetchone()
        assert found == ('YES',)


def test_the_vector_schema_exists_empty_and_the_application_role_cannot_create_in_it(
    pg, catalog
) -> None:
    assert catalog.execute(
        "SELECT count(*) FROM pg_namespace WHERE nspname = 'vec'"
    ).fetchone() == (1,)
    assert catalog.execute(
        "SELECT count(*) FROM pg_class WHERE relnamespace = 'vec'::regnamespace"
    ).fetchone() == (0,)
    app = pg.settings.app_user
    assert catalog.execute(
        "SELECT has_schema_privilege(%s, 'vec', 'USAGE'), has_schema_privilege(%s, 'vec', 'CREATE')",
        (app, app),
    ).fetchone() == (True, False)


def test_every_table_in_the_vector_schema_is_closed_by_forced_row_level_security(catalog) -> None:
    open_tables = catalog.execute(
        "SELECT relname FROM pg_class WHERE relnamespace = 'vec'::regnamespace AND relkind = 'r' "
        'AND NOT (relrowsecurity AND relforcerowsecurity)'
    ).fetchall()
    assert open_tables == []


def test_downgrade_to_0009_removes_what_0010_made_and_the_upgrade_makes_it_again(
    pg: PgDatabase,
) -> None:
    engine = pg.engine('owner')
    cfg = migrate.alembic_config()
    with engine.connect() as conn:
        cfg.attributes['connection'] = conn
        command.downgrade(cfg, '0009_ingest_retry_at')
        conn.commit()
        found = {
            r[0]
            for r in conn.exec_driver_sql(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        }
        assert not (plan.TABLES & found)
        assert not conn.exec_driver_sql(
            "SELECT 1 FROM pg_namespace WHERE nspname = 'vec'"
        ).fetchall()
        assert not conn.exec_driver_sql(
            "SELECT 1 FROM pg_proc WHERE proname IN ('claims_append_only', 'op_erasure_allows')"
        ).fetchall()
        columns = conn.exec_driver_sql(
            "SELECT 1 FROM information_schema.columns WHERE column_name IN ('dataset_name', 'dataset_import_id') "
            "AND table_name IN ('dataset_imports', 'label_operations')"
        ).fetchall()
        assert columns == []
    engine.dispose()
    migrate.upgrade(pg.settings, super_user=pg.server.super_user)
    migrate.check(pg.settings)
    with psycopg.connect(pg.conninfo('super')) as conn:
        found = {
            r[0]
            for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        }
    assert found >= plan.TABLES
