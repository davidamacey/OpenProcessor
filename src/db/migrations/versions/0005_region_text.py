"""Region text: reader runs, consensus groups and members, and the consensus sweep marker.

Revision ID: 0005_region_text
Revises: 0004_privacy_purges

Alembic: partial unique indexes take ``postgresql_where`` and ``postgresql_nulls_not_distinct`` on
``op.create_index`` (https://alembic.sqlalchemy.org/en/latest/ops.html#alembic.operations.Operations.create_index,
https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#partial-indexes); row-level security has
no operation, so the PostgreSQL statements run through ``op.execute``
(https://www.postgresql.org/docs/current/ddl-rowsecurity.html), as in revision 0003.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from src.db.models.region_text import (
    CONSENSUS_BASES,
    CONSENSUS_METHODS,
    CONSENSUS_TIERS,
    RUN_TRIGGERS,
)
from src.db.types import in_list


revision = '0005_region_text'
down_revision = '0004_privacy_purges'
branch_labels = None
depends_on = None

TABLES = ('region_text_runs', 'region_text_consensus_groups', 'region_text_consensus_members')
_MARKER = 'region_text_consensus_marker'
_KEYS_OLD = (
    'cluster_refresh',
    'region_drain',
    'region_dependency_health',
    'vlm_external_ack',
    'open_vocab_gate',
)
_DIRECT = '(SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())'


def _replace_state_key_check(keys: tuple[str, ...]) -> None:
    op.drop_constraint(op.f('ck_project_state_key'), 'project_state', type_='check')
    op.create_check_constraint(op.f('ck_project_state_key'), 'project_state', in_list('key', keys))


def _create_runs() -> None:
    op.create_table(
        'region_text_runs',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column('trigger', sa.Text(), nullable=False),
        sa.Column('reader', sa.Text(), nullable=False),
        sa.Column('reader_version', sa.Text(), nullable=False),
        sa.Column('profile_config_id', sa.Uuid(), nullable=True),
        sa.Column('profile_revision', sa.Integer(), nullable=True),
        sa.Column(
            'params', postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"), nullable=False,
        ),
        sa.Column(
            'counts', postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"), nullable=False,
        ),
        sa.Column(
            'started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            in_list('trigger', RUN_TRIGGERS), name=op.f('ck_region_text_runs_trigger')
        ),
        sa.CheckConstraint(
            '(profile_config_id IS NULL) = (profile_revision IS NULL)',
            name=op.f('ck_region_text_runs_profile_pair'),
        ),
        sa.ForeignKeyConstraint(
            ['project_id'], ['projects.id'],
            name=op.f('fk_region_text_runs_project_id_projects'), ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'],
            name=op.f('fk_region_text_runs_job_id_jobs'), ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['profile_config_id', 'profile_revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_region_text_runs_profile',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_region_text_runs')),
        sa.UniqueConstraint('id', 'project_id', name='uq_region_text_runs_id_project_id'),
    )  # fmt: skip
    op.create_index(
        'ix_region_text_runs_project',
        'region_text_runs',
        ['project_id', sa.text('started_at DESC')],
    )
    op.create_index(
        'uq_region_text_runs_open_pass',
        'region_text_runs',
        ['project_id', 'reader', 'reader_version', 'profile_config_id', 'profile_revision'],
        unique=True,
        postgresql_where=sa.text("trigger = 'region_pass' AND finished_at IS NULL"),
        postgresql_nulls_not_distinct=True,
    )


def _create_groups() -> None:
    op.create_table(
        'region_text_consensus_groups',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('basis', sa.Text(), nullable=False),
        sa.Column('method', sa.Text(), nullable=False),
        sa.Column('n_frames', sa.Integer(), nullable=False),
        sa.Column('n_agree', sa.Integer(), nullable=False),
        sa.Column('confidence', sa.REAL(), nullable=True),
        sa.Column('tier', sa.Text(), nullable=False),
        sa.Column('jurisdiction', sa.Text(), nullable=True),
        sa.Column('jurisdiction_conf', sa.REAL(), nullable=True),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            in_list('basis', CONSENSUS_BASES), name=op.f('ck_region_text_consensus_groups_basis')
        ),
        sa.CheckConstraint(
            in_list('method', CONSENSUS_METHODS),
            name=op.f('ck_region_text_consensus_groups_method'),
        ),
        sa.CheckConstraint(
            in_list('tier', CONSENSUS_TIERS), name=op.f('ck_region_text_consensus_groups_tier')
        ),
        sa.CheckConstraint(
            'n_frames >= 1', name=op.f('ck_region_text_consensus_groups_n_frames')
        ),
        sa.CheckConstraint(
            'n_agree BETWEEN 0 AND n_frames', name=op.f('ck_region_text_consensus_groups_n_agree')
        ),
        sa.CheckConstraint(
            'jurisdiction_conf BETWEEN 0 AND 1',
            name=op.f('ck_region_text_consensus_groups_jurisdiction_conf'),
        ),
        sa.ForeignKeyConstraint(
            ['project_id'], ['projects.id'],
            name=op.f('fk_region_text_consensus_groups_project_id_projects'), ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['run_id', 'project_id'],
            ['region_text_runs.id', 'region_text_runs.project_id'],
            name='fk_region_text_consensus_groups_run_project', ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_region_text_consensus_groups')),
        sa.UniqueConstraint(
            'id', 'project_id', name='uq_region_text_consensus_groups_id_project_id'
        ),
    )  # fmt: skip
    op.create_index(
        'ix_region_text_consensus_groups_run',
        'region_text_consensus_groups',
        ['run_id', 'project_id'],
    )


def _create_members() -> None:
    op.create_table(
        'region_text_consensus_members',
        sa.Column('group_id', sa.Uuid(), nullable=False),
        sa.Column('item_id', sa.Text(), nullable=False),
        sa.Column('box_id', sa.Text(), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('weight', sa.REAL(), nullable=False),
        sa.Column('agrees', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ['project_id'], ['projects.id'],
            name=op.f('fk_region_text_consensus_members_project_id_projects'), ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['group_id', 'project_id'],
            ['region_text_consensus_groups.id', 'region_text_consensus_groups.project_id'],
            name='fk_region_text_consensus_members_group_project', ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint(
            'group_id', 'item_id', 'box_id', name=op.f('pk_region_text_consensus_members')
        ),
        sa.UniqueConstraint(
            'project_id', 'item_id', 'box_id', name='uq_region_text_consensus_members_box'
        ),
    )  # fmt: skip


def _enable_security() -> None:
    for table in TABLES:
        op.execute(f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY')
        op.execute(f'ALTER TABLE {table} FORCE ROW LEVEL SECURITY')
        op.execute(
            f'CREATE POLICY p_{table}_project ON {table} FOR ALL '
            f'USING ({_DIRECT}) WITH CHECK ({_DIRECT})'
        )


def upgrade() -> None:
    _create_runs()
    _create_groups()
    _create_members()
    _enable_security()
    _replace_state_key_check((*_KEYS_OLD, _MARKER))


def downgrade() -> None:
    op.execute(sa.text('DELETE FROM project_state WHERE key = :key').bindparams(key=_MARKER))
    _replace_state_key_check(_KEYS_OLD)
    for table in reversed(TABLES):
        op.drop_table(table)
