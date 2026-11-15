"""The storage record: photos, claims, results, current state and the gallery tables.

Revision ID: 0010_storage_record
Revises: 0009_ingest_retry_at

Implements docs/design/v070_rebuild/15a_storage_schema.md sections 2, 8 and 9 for the tables that
are new: 35 tables, their keys, foreign-key indexes, row-level-security policies, the append-only
trigger on ``claims`` with the erasure guard, the immutable ``analysers`` catalogue and the seeded
``claim_kinds``. Not here, by design: list and sort indexes, summary tables and fillfactors (15a
section 2.6, from gate A); the per-space vector tables (made at run time by the embedding-space job as
``op_vector``); the changes to existing tables that break their current writers
(``embedding_spaces``, ``cluster_memberships``, ``near_dup_*``, ``dataset_import_images``,
``label_operation_members``), which belong to the lanes that rewrite those writers.

Official documentation read for the choices: PostgreSQL 18 identity columns, CREATE TABLE (identity,
generated columns, CHECK, FOREIGN KEY), CREATE POLICY and ALTER TABLE ... FORCE ROW LEVEL SECURITY,
CREATE INDEX (partial, NULLS NOT DISTINCT), CREATE TRIGGER, CREATE EXTENSION and ``trusted``
extensions; Alembic ``op.execute`` and offline SQL generation (the revision is plain DDL, so it renders
the same with ``as_sql``).
"""

from __future__ import annotations

from alembic import op

from src.db.migrations.schema_0010 import ddl, record, results, security, state


revision = '0010_storage_record'
down_revision = '0009_ingest_retry_at'
branch_labels = None
depends_on = None

# 15a section 9: analysers, claim_kinds; photos; items; entities; claims; results; state; the rest
_TABLES = (*record.TABLES, *results.TABLES, *state.TABLES)
_AUTOVACUUM_TABLES = ('item_state', 'box_state')


def _add_columns_to_existing_tables() -> None:
    op.execute('ALTER TABLE dataset_imports ADD COLUMN dataset_name text NULL')
    op.execute('ALTER TABLE label_operations ADD COLUMN dataset_import_id uuid NULL')
    op.execute(
        'ALTER TABLE label_operations ADD CONSTRAINT fk_label_operations_dataset_import_id_dataset_imports '
        'FOREIGN KEY (dataset_import_id) REFERENCES dataset_imports(id) ON DELETE SET NULL'
    )
    op.execute(ddl.foreign_key_index_sql('label_operations', 'dataset_import_id', partial=True))


def upgrade() -> None:
    security.ensure_extensions()
    security.create_vector_schema()
    ddl.create(_TABLES)
    for statement in (*record.INDEXES, *state.INDEXES):
        op.execute(statement)
    for table, column, partial in ddl.FOREIGN_KEY_INDEXES:
        op.execute(ddl.foreign_key_index_sql(table, column, partial=partial))
    for table in _AUTOVACUUM_TABLES:
        op.execute(state.AUTOVACUUM.format(table=table))
    # append only: the visibility map stays set, so index-only scans keep working (15a section 2.4)
    op.execute('ALTER TABLE claims SET (autovacuum_vacuum_insert_scale_factor = 0.01)')
    security.create_policies(_TABLES)
    security.create_functions_and_triggers()
    security.seed_claim_kinds()
    _add_columns_to_existing_tables()


def downgrade() -> None:
    op.execute('ALTER TABLE label_operations DROP COLUMN dataset_import_id')
    op.execute('ALTER TABLE dataset_imports DROP COLUMN dataset_name')
    for table in reversed(_TABLES):
        op.execute(f'DROP TABLE {table.name}')
    security.drop_functions()
    # fails while a vector table exists: a downgrade never drops vectors silently
    op.execute('DROP SCHEMA vec')
