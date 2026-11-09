"""A monotonic sharing revision on ``models`` (it was derived from the consumer count, so a share
followed by an unshare brought it back and a stale writer passed).

Revision ID: 0006_model_sharing_revision
Revises: 0004_privacy_purges
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = '0006_model_sharing_revision'
down_revision = '0004_privacy_purges'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'models', sa.Column('sharing_revision', sa.Integer(), server_default='1', nullable=False)
    )
    # A model that is shared now starts at 2, the value the wire served for it before.
    op.execute(
        'UPDATE models SET sharing_revision = 2 '
        'WHERE EXISTS (SELECT 1 FROM model_shares s WHERE s.model_id = models.id)'
    )
    op.create_check_constraint(
        op.f('ck_models_sharing_revision'), 'models', 'sharing_revision >= 1'
    )


def downgrade() -> None:
    op.drop_constraint(op.f('ck_models_sharing_revision'), 'models', type_='check')
    op.drop_column('models', 'sharing_revision')
