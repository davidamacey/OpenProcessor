"""``ingest_source_files.retry_at``: when a file that failed for a transient reason may be claimed again.

Revision ID: 0009_ingest_retry_at
Revises: 0008_projects_list_index

A nullable column added with no default is stored as a null in the table's metadata, so
``ADD COLUMN`` does not rewrite the table or its indexes (PostgreSQL 18, ALTER TABLE, "ADD
COLUMN": https://www.postgresql.org/docs/18/sql-altertable.html). It takes ``ACCESS EXCLUSIVE``
only for the catalog change; the owner role's ``lock_timeout`` (60 s, ``src/db/roles.py``) ends a
wait behind a long transaction instead of queueing every reader behind it. ``DROP COLUMN`` does
not rewrite either. Alembic ``op.add_column`` / ``op.drop_column``
(https://alembic.sqlalchemy.org/en/latest/ops.html#alembic.operations.Operations.add_column).
No index: the claim already reads the source's rows by the primary key prefix and ``retry_at``
is one more filter on those rows.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = '0009_ingest_retry_at'
down_revision = '0008_projects_list_index'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'ingest_source_files', sa.Column('retry_at', sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('ingest_source_files', 'retry_at')
