"""Tables of revision 0002_research_seams, group 'assist_loop': the auto-label loop record and the hourly VLM usage."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


TABLES = (
    'assist_loop_runs',
    'assist_loop_iterations',
    'assist_loop_nodes',
    'assist_loop_selections',
    'assist_loop_proposals',
    'assist_loop_reviews',
    'assist_loop_budgets',
    'assist_loop_budget_ledger',
    'vlm_usage_hourly',
)


def create() -> None:
    op.create_table(
        'assist_loop_runs',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=False),
        sa.Column('preset_config_id', sa.Uuid(), nullable=True),
        sa.Column('preset_revision', sa.Integer(), nullable=True),
        sa.Column('request', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('strategy', sa.Text(), nullable=False),
        sa.Column('seed', sa.BigInteger(), nullable=False),
        sa.Column('pinned', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('stop_reason', sa.Text(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.Column('created_by_user_id', sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "stop_reason IN ('done', 'budget_crops', 'budget_requests', 'budget_tokens', 'budget_time', 'vlm_failed')",
            name=op.f('ck_assist_loop_runs_stop_reason'),
        ),
        sa.CheckConstraint(
            "strategy IN ('cluster_sampled')", name=op.f('ck_assist_loop_runs_strategy')
        ),
        sa.CheckConstraint(
            '(preset_config_id IS NULL) = (preset_revision IS NULL)',
            name=op.f('ck_assist_loop_runs_preset_pair'),
        ),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'], name=op.f('fk_assist_loop_runs_job_id_jobs')
        ),
        sa.ForeignKeyConstraint(
            ['preset_config_id', 'preset_revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_assist_loop_runs_preset',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_assist_loop_runs_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_assist_loop_runs')),
        sa.UniqueConstraint('job_id', name=op.f('uq_assist_loop_runs_job_id')),
    )
    op.create_index(
        'ix_assist_loop_runs_project',
        'assist_loop_runs',
        ['project_id', sa.literal_column('created_at DESC')],
        unique=False,
    )
    op.create_table(
        'assist_loop_iterations',
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('ordinal', sa.Integer(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column(
            'counts',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            'started_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'closed') = (closed_at IS NOT NULL)",
            name=op.f('ck_assist_loop_iterations_closed_iff_closed_at'),
        ),
        sa.CheckConstraint(
            "status IN ('open', 'closed')", name=op.f('ck_assist_loop_iterations_status')
        ),
        sa.CheckConstraint('ordinal >= 1', name=op.f('ck_assist_loop_iterations_ordinal')),
        sa.ForeignKeyConstraint(
            ['run_id'],
            ['assist_loop_runs.id'],
            name=op.f('fk_assist_loop_iterations_run_id_assist_loop_runs'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('run_id', 'ordinal', name=op.f('pk_assist_loop_iterations')),
    )
    op.create_table(
        'assist_loop_nodes',
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('path', sa.Text(), nullable=False),
        sa.Column('iteration', sa.Integer(), nullable=False),
        sa.Column('parent_path', sa.Text(), nullable=True),
        sa.Column('level', sa.SmallInteger(), nullable=False),
        sa.Column('member_count', sa.Integer(), nullable=False),
        sa.Column('eligible_count', sa.Integer(), nullable=False),
        sa.Column('membership_digest', sa.Text(), nullable=False),
        sa.Column('snapshot_storage_key', sa.Text(), nullable=False),
        sa.Column('step', sa.Text(), nullable=False),
        sa.Column('decision', sa.Text(), nullable=True),
        sa.Column('class_name', sa.Text(), nullable=True),
        sa.Column('answered', sa.Integer(), nullable=True),
        sa.Column('agree', sa.Integer(), nullable=True),
        sa.Column('bound', sa.Float(precision=53), nullable=True),
        sa.Column('distribution', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('admitted_crops', sa.Integer(), nullable=True),
        sa.Column('admitted_requests', sa.Integer(), nullable=True),
        sa.Column('label_op_id', sa.Uuid(), nullable=True),
        sa.Column('items_propagated', sa.Integer(), server_default='0', nullable=False),
        sa.Column(
            'skipped',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(step = 'done') = (decision IS NOT NULL)",
            name=op.f('ck_assist_loop_nodes_done_iff_decided'),
        ),
        sa.CheckConstraint(
            "decision IN ('pass', 'split', 'review_leaf', 'review_small', 'stale', 'unvisited')",
            name=op.f('ck_assist_loop_nodes_decision'),
        ),
        sa.CheckConstraint(
            "path ~ '^r[0-9]+(\\.[0-9]+)*$'", name=op.f('ck_assist_loop_nodes_path')
        ),
        sa.CheckConstraint(
            "step IN ('open', 'admitted', 'asked', 'gated', 'propagating', 'done')",
            name=op.f('ck_assist_loop_nodes_step'),
        ),
        sa.CheckConstraint('eligible_count >= 0', name=op.f('ck_assist_loop_nodes_eligible_count')),
        sa.CheckConstraint('level >= 0', name=op.f('ck_assist_loop_nodes_level')),
        sa.CheckConstraint('member_count >= 0', name=op.f('ck_assist_loop_nodes_member_count')),
        sa.ForeignKeyConstraint(
            ['label_op_id'],
            ['label_operations.id'],
            name=op.f('fk_assist_loop_nodes_label_op_id_label_operations'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['run_id', 'iteration'],
            ['assist_loop_iterations.run_id', 'assist_loop_iterations.ordinal'],
            name='fk_assist_loop_nodes_iteration',
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['run_id'],
            ['assist_loop_runs.id'],
            name=op.f('fk_assist_loop_nodes_run_id_assist_loop_runs'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('run_id', 'path', name=op.f('pk_assist_loop_nodes')),
    )
    op.create_index(
        'ix_assist_loop_nodes_decision',
        'assist_loop_nodes',
        ['run_id', 'decision', 'level'],
        unique=False,
    )
    op.create_index(
        'ix_assist_loop_nodes_open',
        'assist_loop_nodes',
        ['run_id'],
        unique=False,
        postgresql_where=sa.text("step <> 'done'"),
    )
    op.create_table(
        'assist_loop_selections',
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('item_id', sa.Text(), nullable=False),
        sa.Column('purpose', sa.Text(), nullable=False),
        sa.Column('node_path', sa.Text(), nullable=True),
        sa.Column('iteration', sa.Integer(), nullable=False),
        sa.Column('score', sa.Float(precision=53), nullable=True),
        sa.Column('draw_rank', sa.Integer(), nullable=False),
        sa.Column(
            'selected_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "purpose IN ('node_draw', 'node_topup', 'redraw', 'audit', 'blind_audit')",
            name=op.f('ck_assist_loop_selections_purpose'),
        ),
        sa.ForeignKeyConstraint(
            ['run_id'],
            ['assist_loop_runs.id'],
            name=op.f('fk_assist_loop_selections_run_id_assist_loop_runs'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint(
            'run_id', 'item_id', 'purpose', name=op.f('pk_assist_loop_selections')
        ),
    )
    op.create_table(
        'assist_loop_proposals',
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('item_id', sa.Text(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('proposer_kind', sa.Text(), nullable=False),
        sa.Column('proposer_ref', sa.Text(), nullable=False),
        sa.Column('request_id', sa.Uuid(), nullable=False),
        sa.Column('outcome', sa.Text(), nullable=False),
        sa.Column('class_name', sa.Text(), nullable=True),
        sa.Column('new_class_name', sa.Text(), nullable=True),
        sa.Column('attributes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('item_write', sa.Text(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(outcome = 'answered') = (class_name IS NOT NULL OR new_class_name IS NOT NULL)",
            name=op.f('ck_assist_loop_proposals_answered_has_class'),
        ),
        sa.CheckConstraint(
            "item_write IN ('written', 'locked', 'token_changed', 'not_attempted')",
            name=op.f('ck_assist_loop_proposals_item_write'),
        ),
        sa.CheckConstraint(
            "outcome IN ('answered', 'unreadable', 'failed')",
            name=op.f('ck_assist_loop_proposals_outcome'),
        ),
        sa.CheckConstraint(
            "proposer_kind IN ('vlm')", name=op.f('ck_assist_loop_proposals_proposer_kind')
        ),
        sa.CheckConstraint(
            "role IN ('primary', 'audit')", name=op.f('ck_assist_loop_proposals_role')
        ),
        sa.ForeignKeyConstraint(
            ['run_id'],
            ['assist_loop_runs.id'],
            name=op.f('fk_assist_loop_proposals_run_id_assist_loop_runs'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('run_id', 'item_id', 'role', name=op.f('pk_assist_loop_proposals')),
    )
    op.create_index(
        'ix_assist_loop_proposals_request',
        'assist_loop_proposals',
        ['run_id', 'request_id'],
        unique=False,
    )
    op.create_table(
        'assist_loop_reviews',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('item_id', sa.Text(), nullable=False),
        sa.Column('queue', sa.Text(), nullable=False),
        sa.Column('machine_source', sa.Text(), nullable=False),
        sa.Column('machine_class_name', sa.Text(), nullable=True),
        sa.Column('shown_class_name', sa.Text(), nullable=True),
        sa.Column('decision', sa.Text(), nullable=False),
        sa.Column('human_class_name', sa.Text(), nullable=True),
        sa.Column('served_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            'decided_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('active_ms', sa.Integer(), nullable=True),
        sa.Column('batch_size', sa.SmallInteger(), server_default='1', nullable=False),
        sa.Column('label_op_id', sa.Uuid(), nullable=True),
        sa.Column('reviewer', sa.Text(), nullable=False),
        sa.Column('reviewer_user_id', sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "(decision IN ('accept', 'correct')) = (human_class_name IS NOT NULL)",
            name=op.f('ck_assist_loop_reviews_human_class_iff_accepted'),
        ),
        sa.CheckConstraint(
            "decision IN ('accept', 'correct', 'reject', 'skip')",
            name=op.f('ck_assist_loop_reviews_decision'),
        ),
        sa.CheckConstraint(
            "machine_source IN ('propagated', 'direct', 'proposal_only')",
            name=op.f('ck_assist_loop_reviews_machine_source'),
        ),
        sa.CheckConstraint(
            "queue <> 'blind_audit' OR shown_class_name IS NULL",
            name=op.f('ck_assist_loop_reviews_blind_audit_hides_class'),
        ),
        sa.CheckConstraint(
            "queue IN ('verify', 'blind_audit')", name=op.f('ck_assist_loop_reviews_queue')
        ),
        sa.CheckConstraint(
            'active_ms BETWEEN 0 AND 600000', name=op.f('ck_assist_loop_reviews_active_ms')
        ),
        sa.CheckConstraint(
            'batch_size BETWEEN 1 AND 200', name=op.f('ck_assist_loop_reviews_batch_size')
        ),
        sa.ForeignKeyConstraint(
            ['label_op_id'],
            ['label_operations.id'],
            name=op.f('fk_assist_loop_reviews_label_op_id_label_operations'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['run_id'],
            ['assist_loop_runs.id'],
            name=op.f('fk_assist_loop_reviews_run_id_assist_loop_runs'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_assist_loop_reviews')),
        sa.UniqueConstraint('run_id', 'item_id', 'queue', name='uq_assist_loop_reviews_item_queue'),
    )
    op.create_table(
        'assist_loop_budgets',
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('pool', sa.Text(), nullable=False),
        sa.Column('unit', sa.Text(), nullable=False),
        sa.Column('limit_value', sa.BigInteger(), nullable=True),
        sa.Column('reserved', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('spent', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint("pool IN ('vlm', 'audit')", name=op.f('ck_assist_loop_budgets_pool')),
        sa.CheckConstraint(
            "unit = 'prompt_tokens' OR limit_value IS NULL OR spent + reserved <= limit_value",
            name=op.f('ck_assist_loop_budgets_no_over_reserve'),
        ),
        sa.CheckConstraint(
            "unit IN ('crops', 'requests', 'prompt_tokens', 'completion_tokens', 'wall_s')",
            name=op.f('ck_assist_loop_budgets_unit'),
        ),
        sa.CheckConstraint(
            'limit_value IS NULL OR limit_value >= 0',
            name=op.f('ck_assist_loop_budgets_limit_value'),
        ),
        sa.CheckConstraint('reserved >= 0', name=op.f('ck_assist_loop_budgets_reserved')),
        sa.CheckConstraint('spent >= 0', name=op.f('ck_assist_loop_budgets_spent')),
        sa.ForeignKeyConstraint(
            ['run_id'],
            ['assist_loop_runs.id'],
            name=op.f('fk_assist_loop_budgets_run_id_assist_loop_runs'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('run_id', 'pool', 'unit', name='pk_assist_loop_budgets'),
    )
    op.create_table(
        'assist_loop_budget_ledger',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('request_id', sa.Uuid(), nullable=False),
        sa.Column('pool', sa.Text(), nullable=False),
        sa.Column('unit', sa.Text(), nullable=False),
        sa.Column('entry', sa.Text(), nullable=False),
        sa.Column('amount', sa.BigInteger(), nullable=False),
        sa.Column('fencing_token', sa.BigInteger(), nullable=False),
        sa.Column(
            'at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.CheckConstraint(
            "entry IN ('reserve', 'settle', 'release', 'forfeit')",
            name=op.f('ck_assist_loop_budget_ledger_entry'),
        ),
        sa.ForeignKeyConstraint(
            ['run_id', 'pool', 'unit'],
            ['assist_loop_budgets.run_id', 'assist_loop_budgets.pool', 'assist_loop_budgets.unit'],
            name='fk_assist_loop_ledger_budget',
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['run_id'],
            ['assist_loop_runs.id'],
            name=op.f('fk_assist_loop_budget_ledger_run_id_assist_loop_runs'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_assist_loop_budget_ledger')),
    )
    op.create_index(
        'ix_assist_loop_ledger_request',
        'assist_loop_budget_ledger',
        ['run_id', 'request_id'],
        unique=False,
    )
    op.create_table(
        'vlm_usage_hourly',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('endpoint_config_id', sa.Uuid(), nullable=False),
        sa.Column('source', sa.Text(), nullable=False),
        sa.Column('hour', sa.DateTime(timezone=True), nullable=False),
        sa.Column('requests', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('crops', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('prompt_tokens', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('completion_tokens', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('usage_missing', sa.BigInteger(), server_default='0', nullable=False),
        sa.CheckConstraint(
            "hour = date_trunc('hour', hour)", name=op.f('ck_vlm_usage_hourly_hour')
        ),
        sa.CheckConstraint(
            "source IN ('assist_loop', 'assist_run', 'autolabel', 'vlm_worker', 'region_verify', 'label_batch')",
            name=op.f('ck_vlm_usage_hourly_source'),
        ),
        sa.ForeignKeyConstraint(
            ['endpoint_config_id'],
            ['config_documents.id'],
            name=op.f('fk_vlm_usage_hourly_endpoint_config_id_config_documents'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_vlm_usage_hourly_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint(
            'project_id', 'endpoint_config_id', 'source', 'hour', name=op.f('pk_vlm_usage_hourly')
        ),
    )
