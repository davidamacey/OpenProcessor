"""The research-seam tables: serving reports, clustering, the embedding registry, the assist loop.

Revision ID: 0002_research_seams
Revises: 0001_control_plane
"""

from __future__ import annotations

from alembic import op

from src.db.migrations.schema_0002 import assist_loop, clustering, embeddings, security, serving


revision = '0002_research_seams'
down_revision = '0001_control_plane'
branch_labels = None
depends_on = None

_GROUPS = (serving, clustering, embeddings, assist_loop)


def upgrade() -> None:
    for group in _GROUPS:
        group.create()
    security.create()


def downgrade() -> None:
    for group in reversed(_GROUPS):
        for table in reversed(group.TABLES):
            op.drop_table(table)
    security.drop_functions()
