"""The control-plane schema: projects, classes, config store, jobs, events, datasets, training.

Revision ID: 0001_control_plane
Revises:
"""

from __future__ import annotations

from alembic import op

from src.db.migrations.schema_0001 import base, domain, jobs, security


revision = '0001_control_plane'
down_revision = None
branch_labels = None
depends_on = None

_GROUPS = (base, jobs, domain)


def upgrade() -> None:
    for group in _GROUPS:
        group.create()
    security.create()


def downgrade() -> None:
    security.drop_cross_table_policies()
    for group in reversed(_GROUPS):
        for table in reversed(group.TABLES):
            op.drop_table(table)
    security.drop_functions()
