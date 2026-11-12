"""Keyset index for the project list's ``updated_at:desc`` sort.

Revision ID: 0008_projects_list_index
Revises: 0007_region_text

Alembic ``op.create_index`` takes expressions such as ``sa.text('updated_at DESC')``
(https://alembic.sqlalchemy.org/en/latest/ops.html#alembic.operations.Operations.create_index).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = '0008_projects_list_index'
down_revision = '0007_region_text'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_projects_updated_at_slug',
        'projects',
        [sa.text('updated_at DESC'), sa.text('slug DESC')],
    )


def downgrade() -> None:
    op.drop_index('ix_projects_updated_at_slug', table_name='projects')
