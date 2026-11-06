"""Tables of revision 0002_research_seams, group 'serving': serving reports (deployment data, no row-level security)."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


TABLES = ('serving_reports',)


def create() -> None:
    op.create_table(
        'serving_reports',
        sa.Column('server', sa.Text(), nullable=False),
        sa.Column('model', sa.Text(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('state', sa.Text(), nullable=False),
        sa.Column('instances', sa.Integer(), nullable=True),
        sa.Column('device_mib', sa.Integer(), nullable=True),
        sa.Column(
            'seen_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.CheckConstraint(
            "state IN ('ready', 'unavailable', 'loading', 'unloading')",
            name=op.f('ck_serving_reports_state'),
        ),
        sa.PrimaryKeyConstraint('server', 'model', 'version', name=op.f('pk_serving_reports')),
    )
