"""The counts-only record of face and text purges (replaces ``data/audit/*.jsonl``).

Revision ID: 0004_privacy_purges
Revises: 0003_near_duplicates
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from src.db.models.ops import COUNTS_ONLY, PURGE_KINDS
from src.db.types import in_list


revision = '0004_privacy_purges'
down_revision = '0003_near_duplicates'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'privacy_purges',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column(
            'at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.Column('counts', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('actor', sa.Text(), nullable=False),
        sa.Column('host', sa.Text(), nullable=False),
        sa.CheckConstraint(in_list('kind', PURGE_KINDS), name=op.f('ck_privacy_purges_kind')),
        sa.CheckConstraint(COUNTS_ONLY, name=op.f('ck_privacy_purges_counts_only')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_privacy_purges')),
    )
    op.create_index('uq_privacy_purges_kind_at', 'privacy_purges', ['kind', 'at'], unique=True)


def downgrade() -> None:
    op.drop_table('privacy_purges')
