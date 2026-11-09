"""Tables of revision 0003_near_duplicates (research near-duplicates plan 09d 5.1, 02d section 3)."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


TABLES = (
    'image_contents',
    'near_dup_runs',
    'near_dup_groups',
    'near_dup_members',
    'near_dup_edges',
    'near_dup_decisions',
    'near_dup_pending',
    'holdout_near_dup_moves',
)
LEVELS = "level IN ('image', 'item', 'core')"
PURPOSES = "purpose IN ('dedup', 'leak')"


def _project(*, nullable: bool, name: str) -> list[sa.schema.SchemaItem]:
    return [
        sa.Column('project_id', sa.Uuid(), nullable=nullable),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f(f'fk_{name}_project_id_projects'),
            ondelete='CASCADE',
        ),
    ]


def _timestamp(name: str) -> sa.Column:
    return sa.Column(
        name, sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
    )


def _create_contents() -> None:
    op.create_table(
        'image_contents',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('imohash', sa.Text(), nullable=False),
        sa.Column('content_sha256', sa.Text(), nullable=True),
        sa.Column('image_id', sa.Text(), nullable=False),
        sa.Column('image_path', sa.Text(), nullable=False),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        _timestamp('created_at'),
        *_project(nullable=True, name='image_contents'),
        sa.CheckConstraint("imohash ~ '^[0-9a-f]{32}$'", name=op.f('ck_image_contents_imohash')),
        sa.CheckConstraint(
            "content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f('ck_image_contents_sha256'),
        ),
        sa.CheckConstraint('size_bytes >= 0', name=op.f('ck_image_contents_size')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_image_contents')),
        sa.UniqueConstraint(
            'project_id', 'image_id', name='uq_image_contents_image',
            postgresql_nulls_not_distinct=True,
        ),
    )  # fmt: skip
    op.create_index(
        'ix_image_contents_imohash', 'image_contents', ['project_id', 'imohash'], unique=False
    )
    op.create_index(
        'uq_image_contents_sha256',
        'image_contents',
        ['project_id', 'content_sha256'],
        unique=True,
        postgresql_where=sa.text('content_sha256 IS NOT NULL'),
        postgresql_nulls_not_distinct=True,
    )


def _create_runs() -> None:
    op.create_table(
        'near_dup_runs',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('level', sa.Text(), nullable=False),
        sa.Column('mode', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column('parent_run_id', sa.Uuid(), nullable=True),
        sa.Column('profile_config_id', sa.Uuid(), nullable=True),
        sa.Column('profile_revision', sa.Integer(), nullable=True),
        sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('space', sa.Text(), nullable=False),
        sa.Column('algorithm_version', sa.Text(), nullable=False),
        *[
            sa.Column(c, sa.Integer(), server_default='0', nullable=False)
            for c in (
                'members_scored', 'dedup_groups', 'leak_groups', 'grouped_members',
                'largest_group', 'oversized_split', 'holdout_absorbed',
            )
        ],
        _timestamp('started_at'),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        *_project(nullable=True, name='near_dup_runs'),
        sa.CheckConstraint(LEVELS, name=op.f('ck_near_dup_runs_level')),
        sa.CheckConstraint("mode IN ('full', 'incremental')", name=op.f('ck_near_dup_runs_mode')),
        sa.CheckConstraint(
            "status IN ('building', 'current', 'superseded', 'failed')",
            name=op.f('ck_near_dup_runs_status'),
        ),
        sa.CheckConstraint(
            "(level = 'core') = (project_id IS NULL)", name=op.f('ck_near_dup_runs_core_has_no_project')
        ),
        sa.CheckConstraint(
            "(level = 'core') = (profile_config_id IS NULL)",
            name=op.f('ck_near_dup_runs_core_has_no_profile'),
        ),
        sa.CheckConstraint(
            '(profile_config_id IS NULL) = (profile_revision IS NULL)',
            name=op.f('ck_near_dup_runs_profile_pair'),
        ),
        sa.CheckConstraint(
            "(status = 'building') = (finished_at IS NULL)",
            name=op.f('ck_near_dup_runs_building_unfinished'),
        ),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'], name=op.f('fk_near_dup_runs_job_id_jobs'), ondelete='SET NULL'
        ),
        sa.ForeignKeyConstraint(
            ['parent_run_id'], ['near_dup_runs.id'],
            name=op.f('fk_near_dup_runs_parent_run'), ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['profile_config_id', 'profile_revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_near_dup_runs_profile',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_near_dup_runs')),
        sa.UniqueConstraint('id', 'project_id', name='uq_near_dup_runs_id_project_id'),
    )  # fmt: skip
    for name, status in (('current', 'current'), ('building', 'building')):
        op.create_index(
            f'uq_near_dup_runs_{name}',
            'near_dup_runs',
            ['project_id', 'level'],
            unique=True,
            postgresql_where=sa.text(f"status = '{status}'"),
            postgresql_nulls_not_distinct=True,
        )
    op.create_index(
        'ix_near_dup_runs_recent',
        'near_dup_runs',
        ['project_id', 'level', sa.literal_column('started_at DESC')],
        unique=False,
    )


def _run_child_keys(table: str) -> list[sa.schema.SchemaItem]:
    return [
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(
            ['run_id', 'project_id'],
            ['near_dup_runs.id', 'near_dup_runs.project_id'],
            name=f'fk_{table}_run_project',
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['run_id'],
            ['near_dup_runs.id'],
            name=op.f(f'fk_{table}_run_id_near_dup_runs'),
            ondelete='CASCADE',
        ),
    ]


def _create_groups() -> None:
    op.create_table(
        'near_dup_groups',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('purpose', sa.Text(), nullable=False),
        sa.Column('lineage_id', sa.Uuid(), nullable=False),
        sa.Column('representative_id', sa.Text(), nullable=False),
        sa.Column('member_count', sa.Integer(), nullable=False),
        sa.Column('min_edge_cosine', sa.REAL(), nullable=False),
        sa.Column('min_cosine_to_rep', sa.REAL(), nullable=False),
        sa.Column('has_holdout', sa.Boolean(), server_default='false', nullable=False),
        *_run_child_keys('near_dup_groups'),
        sa.CheckConstraint(PURPOSES, name=op.f('ck_near_dup_groups_purpose')),
        sa.CheckConstraint('member_count >= 2', name=op.f('ck_near_dup_groups_size')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_near_dup_groups')),
        sa.UniqueConstraint('id', 'project_id', name='uq_near_dup_groups_id_project_id'),
    )  # fmt: skip
    op.create_index(
        'ix_near_dup_groups_run',
        'near_dup_groups',
        ['run_id', 'purpose', sa.literal_column('member_count DESC')],
        unique=False,
    )
    op.create_index('ix_near_dup_groups_lineage', 'near_dup_groups', ['lineage_id'], unique=False)
    op.create_table(
        'near_dup_members',
        sa.Column('purpose', sa.Text(), nullable=False),
        sa.Column('member_id', sa.Text(), nullable=False),
        sa.Column('group_id', sa.Uuid(), nullable=False),
        sa.Column('is_representative', sa.Boolean(), nullable=False),
        sa.Column('cosine_to_rep', sa.REAL(), nullable=False),
        *_run_child_keys('near_dup_members'),
        sa.CheckConstraint(PURPOSES, name=op.f('ck_near_dup_members_purpose')),
        sa.ForeignKeyConstraint(
            ['group_id'], ['near_dup_groups.id'],
            name=op.f('fk_near_dup_members_group_id_near_dup_groups'), ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('run_id', 'purpose', 'member_id', name=op.f('pk_near_dup_members')),
    )  # fmt: skip
    op.create_index('ix_near_dup_members_group', 'near_dup_members', ['group_id'], unique=False)
    op.create_index(
        'uq_near_dup_members_rep',
        'near_dup_members',
        ['group_id'],
        unique=True,
        postgresql_where=sa.text('is_representative'),
    )


def _create_edges_and_state() -> None:
    op.create_table(
        'near_dup_edges',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('level', sa.Text(), nullable=False),
        sa.Column('space', sa.Text(), nullable=False),
        sa.Column('a_id', sa.Text(), nullable=False),
        sa.Column('b_id', sa.Text(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('score', sa.REAL(), nullable=False),
        _timestamp('computed_at'),
        *_project(nullable=True, name='near_dup_edges'),
        sa.CheckConstraint(LEVELS, name=op.f('ck_near_dup_edges_level')),
        sa.CheckConstraint("kind IN ('embedding')", name=op.f('ck_near_dup_edges_kind')),
        sa.CheckConstraint('a_id < b_id', name=op.f('ck_near_dup_edges_ordered')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_near_dup_edges')),
        sa.UniqueConstraint(
            'project_id', 'level', 'space', 'kind', 'a_id', 'b_id', name='uq_near_dup_edges_pair',
            postgresql_nulls_not_distinct=True,
        ),
    )  # fmt: skip
    op.create_index(
        'ix_near_dup_edges_b',
        'near_dup_edges',
        ['project_id', 'level', 'space', 'b_id'],
        unique=False,
    )
    op.create_table(
        'near_dup_decisions',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('level', sa.Text(), nullable=False),
        sa.Column('a_id', sa.Text(), nullable=False),
        sa.Column('b_id', sa.Text(), nullable=False),
        sa.Column('decision', sa.Text(), nullable=False),
        sa.Column('decided_by', sa.Text(), nullable=False),
        sa.Column('decided_by_user_id', sa.Uuid(), nullable=True),
        _timestamp('decided_at'),
        *_project(nullable=False, name='near_dup_decisions'),
        sa.CheckConstraint("level IN ('image', 'item')", name=op.f('ck_near_dup_decisions_level')),
        sa.CheckConstraint(
            "decision IN ('same', 'different')", name=op.f('ck_near_dup_decisions_decision')
        ),
        sa.CheckConstraint('a_id < b_id', name=op.f('ck_near_dup_decisions_ordered')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_near_dup_decisions')),
        sa.UniqueConstraint(
            'project_id', 'level', 'a_id', 'b_id', name='uq_near_dup_decisions_pair'
        ),
    )  # fmt: skip
    op.create_table(
        'near_dup_pending',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('level', sa.Text(), nullable=False),
        sa.Column('member_id', sa.Text(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        _timestamp('enqueued_at'),
        sa.Column('claimed_by_job_id', sa.Uuid(), nullable=True),
        *_project(nullable=True, name='near_dup_pending'),
        sa.CheckConstraint(LEVELS, name=op.f('ck_near_dup_pending_level')),
        sa.CheckConstraint(
            "reason IN ('added', 'vector_changed', 'boxes_changed', 'deleted', 'group_changed')",
            name=op.f('ck_near_dup_pending_reason'),
        ),
        sa.ForeignKeyConstraint(
            ['claimed_by_job_id'], ['jobs.id'],
            name=op.f('fk_near_dup_pending_claimed_by_job_id_jobs'), ondelete='SET NULL',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_near_dup_pending')),
        sa.UniqueConstraint(
            'project_id', 'level', 'member_id', name='uq_near_dup_pending_member',
            postgresql_nulls_not_distinct=True,
        ),
    )  # fmt: skip
    op.create_index(
        'ix_near_dup_pending_age',
        'near_dup_pending',
        ['project_id', 'level', 'enqueued_at'],
        unique=False,
    )
    op.create_table(
        'holdout_near_dup_moves',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('item_id', sa.Text(), nullable=False),
        sa.Column('anchor_item_id', sa.Text(), nullable=False),
        sa.Column('group_id', sa.Uuid(), nullable=True),
        sa.Column('freeze_id', sa.Uuid(), nullable=True),
        sa.Column('run_id', sa.Uuid(), nullable=True),
        _timestamp('moved_at'),
        *_project(nullable=False, name='holdout_near_dup_moves'),
        sa.ForeignKeyConstraint(
            ['freeze_id'], ['holdout_freezes.id'],
            name=op.f('fk_holdout_near_dup_moves_freeze_id_holdout_freezes'), ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['run_id'], ['near_dup_runs.id'],
            name=op.f('fk_holdout_near_dup_moves_run_id_near_dup_runs'), ondelete='SET NULL',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_holdout_near_dup_moves')),
        sa.UniqueConstraint('project_id', 'item_id', name='uq_holdout_near_dup_moves_item'),
    )  # fmt: skip


def create() -> None:
    _create_contents()
    _create_runs()
    _create_groups()
    _create_edges_and_state()
