"""Revision 0010_storage_record rendered to SQL with no database, against the hand-written plan.

The expectations are in ``_storage_plan.py`` (written from 15a), never read from the models or the
migration. What only a server can answer (that the SQL runs, that the policies refuse) is in
``test_storage_schema_matches_plan.py`` and the other ``db`` tests of this change.
"""

from __future__ import annotations

import re

import _storage_plan as plan
import pytest
from _storage_render import (
    create_table_columns,
    create_tables,
    downgrade_sql,
    revision_module,
    statements,
    upgrade_sql,
)

from src.db.base import Base
from src.domain.claim_kinds import KINDS


def test_the_revision_follows_the_current_head() -> None:
    module = revision_module()
    assert module.revision == '0010_storage_record'
    assert module.down_revision == '0009_ingest_retry_at'


def test_every_table_of_the_plan_is_created_once() -> None:
    sql = upgrade_sql()
    created = re.findall(r'^CREATE TABLE (\w+) \(', sql, flags=re.M)
    assert sorted(created) == sorted(plan.TABLES)
    assert len(plan.TABLES) == 35


@pytest.mark.parametrize('table', sorted(plan.TABLES))
def test_each_table_has_the_planned_columns_in_order(table: str) -> None:
    columns = create_table_columns(create_tables(upgrade_sql())[table])
    assert columns == list(plan.COLUMNS[table])


@pytest.mark.parametrize('table', sorted(plan.TABLES - plan.UNPROTECTED))
def test_each_project_table_is_closed_by_row_level_security(table: str) -> None:
    sql = upgrade_sql()
    assert f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY' in sql
    assert f'ALTER TABLE {table} FORCE ROW LEVEL SECURITY' in sql
    policy = re.search(
        rf'CREATE POLICY p_{table}_project ON {table} FOR ALL USING \((.*?)\) WITH CHECK \((.*?)\);',
        sql,
    )
    assert policy, f'no policy on {table}'
    # the InitPlan form of 02a 1.4: each scope function is a scalar subquery
    expected = '(SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())'
    assert policy.group(1) == policy.group(2) == expected


@pytest.mark.parametrize('table', sorted(plan.UNPROTECTED))
def test_a_deployment_catalog_has_no_policy(table: str) -> None:
    assert f'CREATE POLICY p_{table}_project' not in upgrade_sql()
    assert f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY' not in upgrade_sql()
    assert 'project_id' not in create_table_columns(create_tables(upgrade_sql())[table])


def test_the_policy_set_is_exactly_the_plans() -> None:
    found = set(re.findall(r'CREATE POLICY (p_\w+_project) ON (\w+)', upgrade_sql()))
    assert {(t, p) for p, t in found} == plan.POLICIES


def test_the_claim_log_is_append_only_with_the_erasure_guard() -> None:
    sql = upgrade_sql()
    assert 'CREATE TRIGGER claims_append_only BEFORE UPDATE OR DELETE ON claims' in sql
    body = re.search(r'CREATE FUNCTION claims_append_only\(\).*?END \$\$', sql, flags=re.S)
    assert body
    text = body.group(0)
    assert "TG_OP = 'DELETE'" in text
    assert 'op_project_delete_allows(OLD.project_id)' in text
    assert 'op_erasure_allows(OLD.project_id)' in text
    assert 'NOT EXISTS (SELECT 1 FROM projects p WHERE p.id = OLD.project_id)' in text
    assert 'RAISE EXCEPTION' in text


def test_the_erasure_guard_names_a_running_job_of_the_project() -> None:
    sql = upgrade_sql()
    body = re.search(r'CREATE FUNCTION op_erasure_allows\(pid uuid\).*?\$\$;', sql, flags=re.S)
    assert body
    text = body.group(0)
    assert "current_setting('op.erasure_job', true)" in text
    assert "e.state = 'running'" in text
    assert 'e.project_id = pid' in text


def test_analysers_are_immutable() -> None:
    sql = upgrade_sql()
    assert 'CREATE TRIGGER analysers_immutable BEFORE UPDATE OR DELETE ON analysers' in sql


def test_the_trigger_and_function_sets_are_exactly_the_plans() -> None:
    sql = upgrade_sql()
    assert set(
        re.findall(r'CREATE TRIGGER (\w+) \w+ (?:UPDATE|INSERT|DELETE)[^;]*? ON (\w+)', sql)
    ) == {(name, table) for table, name in plan.TRIGGERS}
    assert set(re.findall(r'CREATE FUNCTION (\w+)\(', sql)) == plan.FUNCTIONS


@pytest.mark.parametrize('name', sorted(plan.UNIQUE_CONSTRAINTS))
def test_each_planned_unique_constraint_exists(name: str) -> None:
    assert f'CONSTRAINT {name} UNIQUE' in upgrade_sql()


@pytest.mark.parametrize('name', sorted(plan.PARTIAL_INDEXES | plan.PLAIN_INDEXES))
def test_each_planned_index_exists(name: str) -> None:
    sql = upgrade_sql()
    found = re.search(rf'CREATE (?:UNIQUE )?INDEX {name} ON [^;]*;', sql)
    assert found, name
    if name in plan.NULLS_NOT_DISTINCT:
        assert 'NULLS NOT DISTINCT' in found.group(0)
    if name in plan.PARTIAL_INDEXES:
        assert ' WHERE ' in found.group(0)


def test_the_withdraw_and_operation_keys_are_the_plans() -> None:
    sql = upgrade_sql()
    assert re.search(
        r'CREATE UNIQUE INDEX uq_claims_withdraws ON claims \(withdraws_claim_id\) '
        r'WHERE withdraws_claim_id IS NOT NULL;',
        sql,
    )
    op_key = re.search(
        r'CREATE UNIQUE INDEX uq_claims_op_subject ON claims \(([^)]*)\)([^;]*);', sql
    )
    assert op_key
    assert [c.strip() for c in op_key.group(1).split(',')] == [
        'op_id', 'kind', 'photo_key', 'subject_crop_id', 'subject_box_id', 'subject_face_id',
        'subject_pair_id', 'subject_video_key', 'subject_entity_key',
    ]  # fmt: skip
    assert 'NULLS NOT DISTINCT' in op_key.group(2)
    assert "WHERE op_id IS NOT NULL AND kind <> 'withdraw'" in op_key.group(2)


def test_the_foreign_key_indexes_are_exactly_the_plans() -> None:
    found = set(
        re.findall(
            r'CREATE INDEX (ix_\w+) ON (\w+) \((\w+)\)( WHERE \w+ IS NOT NULL)?;', upgrade_sql()
        )
    )
    fk = {(n, t, c, bool(w)) for n, t, c, w in found if n == f'ix_{t}_{c}'}
    expected = plan.FOREIGN_KEY_INDEXES | plan.ADDED_FOREIGN_KEY_INDEXES
    assert fk == {(f'ix_{t}_{c}', t, c, partial) for t, c, partial in expected}


def test_a_foreign_key_names_what_it_references() -> None:
    sql = upgrade_sql()
    found = set()
    for table, statement in create_tables(sql).items():
        for col, ref, action in re.findall(
            r'FOREIGN KEY \((\w+)\) REFERENCES (\w+)\(\w+\)(?: ON DELETE (CASCADE|SET NULL))?',
            statement,
        ):
            found.add((table, col, ref, {'CASCADE': 'c', 'SET NULL': 'n', '': 'a'}[action]))
        for cols, ref in re.findall(
            r'FOREIGN KEY \(([\w, ]+,[\w, ]+)\) REFERENCES (\w+)', statement
        ):
            # a composite key is counted by its first column, as the catalog test reads it
            found.add((table, cols.split(',')[0].strip(), ref, 'a'))
    assert found == plan.FOREIGN_KEYS


@pytest.mark.parametrize(('table', 'column'), sorted(plan.IDENTITY_KEYS.items()))
def test_a_high_volume_key_is_a_bigint_identity(table: str, column: str) -> None:
    statement = create_tables(upgrade_sql())[table]
    assert re.search(rf'\b{column} bigint GENERATED ALWAYS AS IDENTITY', statement)


def test_claim_kinds_are_seeded_from_the_domain_catalogue() -> None:
    rows = set()
    for s in statements(upgrade_sql()):
        if s.startswith('INSERT INTO claim_kinds'):
            rows.update(re.findall(r"\('(\w+)', '(\w+)', '(\w+)', (true|false), (true|false)\)", s))
    expected = {
        (
            k.value,
            i.subject_kind.value,
            i.layer.value,
            str(i.pins_geometry).lower(),
            str(i.asserts_object).lower(),
        )
        for k, i in KINDS.items()
    }
    assert rows == expected
    assert len(rows) == len(KINDS) == 23


def test_the_extensions_are_asked_for_with_a_message_naming_what_is_missing() -> None:
    sql = upgrade_sql()
    for extension in ('vector', 'pg_trgm'):
        assert f'CREATE EXTENSION IF NOT EXISTS {extension}' in sql
    assert 'RAISE WARNING' in sql
    assert 'superuser' in sql
    assert 'CREATE SCHEMA vec' in sql
    # no vector table is made by the migration: the embedding-space job makes them at run time
    assert 'vec.v' not in sql
    assert 'halfvec' not in sql


def test_downgrade_drops_every_table_before_its_parents_and_every_function() -> None:
    sql = downgrade_sql()
    dropped = re.findall(r'^DROP TABLE (\w+);', sql, flags=re.M)
    assert sorted(dropped) == sorted(plan.TABLES)
    created = re.findall(r'^CREATE TABLE (\w+) \(', upgrade_sql(), flags=re.M)
    assert dropped == list(reversed(created))
    for function in plan.FUNCTIONS:
        assert re.search(rf'DROP FUNCTION {function}\(', sql)
    assert 'DROP SCHEMA vec' in sql


def test_every_modelled_table_agrees_with_the_plan() -> None:
    tables = Base.metadata.tables
    assert set(tables) >= plan.TABLES
    for name, columns in plan.COLUMNS.items():
        assert [c.name for c in tables[name].columns] == list(columns), name


def test_the_model_foreign_key_indexes_are_the_plans() -> None:
    names = set()
    for name in plan.TABLES:
        for index in Base.metadata.tables[name].indexes:
            if (
                index.name == f'ix_{name}_{next(iter(index.columns)).name}'
                and len(index.columns) == 1
            ):
                names.add(index.name)
    assert names == plan.FOREIGN_KEY_INDEX_NAMES


def test_the_model_foreign_keys_are_the_plans() -> None:
    found = set()
    action = {'CASCADE': 'c', 'SET NULL': 'n', None: 'a', 'NO ACTION': 'a'}
    for name in plan.TABLES:
        for fk in Base.metadata.tables[name].foreign_key_constraints:
            element = fk.elements[0]
            found.add((name, element.parent.name, element.column.table.name, action[fk.ondelete]))
    assert found == plan.FOREIGN_KEYS


def test_the_existing_tables_gain_only_the_planned_columns() -> None:
    sql = upgrade_sql()
    added = re.findall(r'^ALTER TABLE (\w+) ADD COLUMN (\w+) ', sql, flags=re.M)
    assert dict(added) == plan.ADDED_COLUMNS
    assert (
        'ALTER TABLE label_operations ADD CONSTRAINT fk_label_operations_dataset_import_id_dataset_imports'
        in sql
    )
    assert (
        'ON DELETE SET NULL'
        in sql.split('fk_label_operations_dataset_import_id_dataset_imports')[1].split(';')[0]
    )
    # nothing that exists is dropped or altered in place
    assert not re.search(r'DROP (TABLE|COLUMN)|ALTER COLUMN|RENAME', sql)
