"""Near-duplicate tables, the holdout freeze columns, the near_dup profile kind and axis.

Revision ID: 0003_near_duplicates
Revises: 0002_research_seams
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from src.db.migrations.schema_0003 import security, tables


revision = '0003_near_duplicates'
down_revision = '0002_research_seams'
branch_labels = None
depends_on = None

_KINDS_OLD = (
    "'prompt_pack', 'region_profile', 'open_vocab_set', 'ingest_profile', 'vlm_endpoint', "
    "'serving_profile', 'cluster_profile', 'assist_loop_preset'"
)
_AXES_OLD = (
    "'prompt_pack', 'detection_profile', 'vlm', 'open_vocab', 'ingest_profile', "
    "'clustering_item', 'clustering_region_box', 'assist_loop_preset'"
)
_PAIRS_OLD = (
    "('prompt_pack', 'prompt_pack'), ('detection_profile', 'region_profile'), "
    "('vlm', 'vlm_endpoint'), ('open_vocab', 'open_vocab_set'), "
    "('ingest_profile', 'ingest_profile'), ('clustering_item', 'cluster_profile'), "
    "('clustering_region_box', 'cluster_profile'), ('assist_loop_preset', 'assist_loop_preset')"
)
_TASK_KINDS_OLD = "'class_rename', 'class_merge', 'cluster_partition'"


def _replace_check(table: str, name: str, expression: str) -> None:
    op.drop_constraint(op.f(name), table, type_='check')
    op.create_check_constraint(op.f(name), table, expression)


def _alter_closed_sets(*, kinds: str, axes: str, pairs: str, task_kinds: str) -> None:
    _replace_check('config_documents', 'ck_config_documents_kind', f'kind IN ({kinds})')
    _replace_check('config_activations', 'ck_config_activations_axis', f'axis IN ({axes})')
    _replace_check(
        'config_activations', 'ck_config_activations_axis_kind', f'(axis, kind) IN ({pairs})'
    )
    _replace_check(
        'search_projection_tasks', 'ck_search_projection_tasks_kind', f'kind IN ({task_kinds})'
    )


def upgrade() -> None:
    tables.create()
    security.create()
    op.add_column(
        'holdout_freezes',
        sa.Column('near_dup_image_run_id', sa.Uuid(), nullable=True),
    )
    op.add_column(
        'holdout_freezes',
        sa.Column('near_dup_item_run_id', sa.Uuid(), nullable=True),
    )
    op.add_column(
        'holdout_freezes',
        sa.Column('moved_by_near_dup', sa.Integer(), server_default='0', nullable=False),
    )
    for column in ('near_dup_image_run_id', 'near_dup_item_run_id'):
        op.create_foreign_key(
            op.f(f'fk_holdout_freezes_{column}_near_dup_runs'),
            'holdout_freezes',
            'near_dup_runs',
            [column],
            ['id'],
            ondelete='SET NULL',
        )
    _alter_closed_sets(
        kinds=f"{_KINDS_OLD}, 'near_dup_profile'",
        axes=f"{_AXES_OLD}, 'near_dup'",
        pairs=f"{_PAIRS_OLD}, ('near_dup', 'near_dup_profile')",
        task_kinds=f"{_TASK_KINDS_OLD}, 'near_dup_run'",
    )


def downgrade() -> None:
    op.execute("DELETE FROM search_projection_tasks WHERE kind = 'near_dup_run'")
    op.execute("DELETE FROM config_activations WHERE axis = 'near_dup'")
    op.execute("DELETE FROM config_documents WHERE kind = 'near_dup_profile'")
    _alter_closed_sets(
        kinds=_KINDS_OLD, axes=_AXES_OLD, pairs=_PAIRS_OLD, task_kinds=_TASK_KINDS_OLD
    )
    for column in ('moved_by_near_dup', 'near_dup_item_run_id', 'near_dup_image_run_id'):
        op.drop_column('holdout_freezes', column)
    for table in reversed(tables.TABLES):
        op.drop_table(table)
    security.drop_function()
