"""Tables of the control-plane schema, group 'jobs', in creation order."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from src.db.types import Xid8


TABLES = (
    'jobs',
    'job_attempts',
    'job_items',
    'events',
    'stream_worker_claims',
    'worker_capabilities',
    'gpu_devices',
    'gpu_reservations',
    'search_projection_tasks',
    'projection_states',
)


def create() -> None:
    op.create_table(
        'jobs',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('scope', sa.Text(), server_default='', nullable=False),
        sa.Column('exclusive_group', sa.Text(), nullable=True),
        sa.Column('queue', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column(
            'params',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column('checkpoint', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            'progress',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column('progress_updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_code', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('attempt', sa.Integer(), server_default='0', nullable=False),
        sa.Column('retry_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('max_attempts', sa.Integer(), server_default='3', nullable=False),
        sa.Column('auto_resume', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('fencing_token', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('owner', sa.Text(), nullable=True),
        sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('max_queue_wait', sa.Interval(), nullable=False),
        sa.Column('dispatch_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('last_dispatched_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancel_requested_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancel_requested_by', sa.Text(), nullable=True),
        sa.Column('pause_requested_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('pause_reason', sa.Text(), nullable=True),
        sa.Column('idempotency_key', sa.Text(), nullable=True),
        sa.Column('parent_job_id', sa.Uuid(), nullable=True),
        sa.Column('requested_by', sa.Text(), nullable=False),
        sa.Column('requested_by_user_id', sa.Uuid(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('queued_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(status IN ('cancelled', 'completed', 'completed_with_errors', 'failed')) = (finished_at IS NOT NULL)",
            name=op.f('ck_jobs_finished_iff_terminal'),
        ),
        sa.CheckConstraint(
            "pause_reason IN ('user', 'audit', 'budget', 'project_paused', 'vlm_unavailable')",
            name=op.f('ck_jobs_pause_reason'),
        ),
        sa.CheckConstraint(
            "status <> 'paused' OR pause_reason IS NOT NULL", name=op.f('ck_jobs_paused_has_reason')
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'paused', 'cancelling', 'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted')",
            name=op.f('ck_jobs_status'),
        ),
        sa.CheckConstraint(
            "status NOT IN ('running', 'cancelling') OR (owner IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name=op.f('ck_jobs_live_has_lease'),
        ),
        sa.CheckConstraint(
            'length(idempotency_key) BETWEEN 8 AND 200', name=op.f('ck_jobs_idempotency_key')
        ),
        sa.CheckConstraint('max_attempts BETWEEN 1 AND 10', name=op.f('ck_jobs_max_attempts')),
        sa.ForeignKeyConstraint(
            ['parent_job_id'],
            ['jobs.id'],
            name=op.f('fk_jobs_parent_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_jobs_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_jobs')),
        postgresql_with={
            'fillfactor': 70,
            'autovacuum_vacuum_scale_factor': 0.02,
            'autovacuum_vacuum_threshold': 50,
            'autovacuum_vacuum_cost_delay': 2,
        },
    )
    op.create_index(
        'ix_jobs_interrupted',
        'jobs',
        ['updated_at'],
        unique=False,
        postgresql_where=sa.text("status = 'interrupted' AND auto_resume"),
    )
    op.create_index(
        'ix_jobs_live',
        'jobs',
        ['status'],
        unique=False,
        postgresql_where=sa.text("status IN ('running', 'cancelling')"),
    )
    op.create_index(
        'ix_jobs_project_recent',
        'jobs',
        ['project_id', 'kind', sa.literal_column('created_at DESC')],
        unique=False,
    )
    op.create_index(
        'ix_jobs_queued',
        'jobs',
        ['queued_at'],
        unique=False,
        postgresql_where=sa.text("status = 'queued'"),
    )
    op.create_index(
        'uq_jobs_idempotency',
        'jobs',
        ['project_id', 'kind', 'idempotency_key'],
        unique=True,
        postgresql_nulls_not_distinct=True,
        postgresql_where=sa.text('idempotency_key IS NOT NULL'),
    )
    op.create_index(
        'uq_jobs_one_active',
        'jobs',
        ['project_id', 'exclusive_group', 'scope'],
        unique=True,
        postgresql_nulls_not_distinct=True,
        postgresql_where=sa.text(
            "exclusive_group IS NOT NULL AND status IN ('cancelling', 'paused', 'queued', 'running')"
        ),
    )
    op.create_table(
        'job_attempts',
        sa.Column('job_id', sa.Uuid(), nullable=False),
        sa.Column('attempt', sa.Integer(), nullable=False),
        sa.Column('fencing_token', sa.BigInteger(), nullable=False),
        sa.Column('owner', sa.Text(), nullable=False),
        sa.Column(
            'started_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('outcome', sa.Text(), nullable=True),
        sa.Column('error_code', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.CheckConstraint(
            "outcome IN ('completed', 'completed_with_errors', 'failed', 'cancelled', 'paused', 'interrupted', 'lost', 'time_limit', 'yielded')",
            name=op.f('ck_job_attempts_outcome'),
        ),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'], name=op.f('fk_job_attempts_job_id_jobs'), ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('job_id', 'attempt', name=op.f('pk_job_attempts')),
    )
    op.create_table(
        'job_items',
        sa.Column('job_id', sa.Uuid(), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('item_id', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'], name=op.f('fk_job_items_job_id_jobs'), ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('job_id', 'position', name=op.f('pk_job_items')),
        sa.UniqueConstraint('job_id', 'item_id', name='uq_job_items_job_id_item_id'),
    )
    op.create_table(
        'events',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('xid', Xid8(), server_default=sa.text('pg_current_xact_id()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            'octet_length(payload::text) <= 7000', name=op.f('ck_events_payload_size')
        ),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'], name=op.f('fk_events_job_id_jobs'), ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_events_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_events')),
        postgresql_with={
            'autovacuum_vacuum_scale_factor': 0.05,
            'autovacuum_vacuum_insert_scale_factor': 0.05,
        },
    )
    op.create_index('ix_events_created', 'events', ['created_at'], unique=False)
    op.create_index('ix_events_cursor', 'events', ['project_id', 'xid', 'id'], unique=False)
    op.create_index(
        'ix_events_cursor_global',
        'events',
        ['xid', 'id'],
        unique=False,
        postgresql_where=sa.text('project_id IS NULL'),
    )
    op.create_table(
        'stream_worker_claims',
        sa.Column('worker_kind', sa.Text(), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('worker_id', sa.Text(), nullable=False),
        sa.Column('fencing_token', sa.BigInteger(), server_default='1', nullable=False),
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "worker_kind IN ('detection', 'vlm')", name=op.f('ck_stream_worker_claims_worker_kind')
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_stream_worker_claims_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('worker_kind', 'project_id', name='pk_stream_worker_claims'),
    )
    op.create_table(
        'worker_capabilities',
        sa.Column('worker_id', sa.Text(), nullable=False),
        sa.Column('queue', sa.Text(), nullable=False),
        sa.Column('host', sa.Text(), nullable=False),
        sa.Column('capabilities', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            'seen_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.PrimaryKeyConstraint('worker_id', name=op.f('pk_worker_capabilities')),
    )
    op.create_table(
        'gpu_devices',
        sa.Column('gpu_uuid', sa.Text(), nullable=False),
        sa.Column('host', sa.Text(), nullable=False),
        sa.Column('nvml_index', sa.Integer(), nullable=False),
        sa.Column('cuda_index', sa.Integer(), nullable=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('compute_capability', sa.Text(), nullable=False),
        sa.Column('memory_total_mib', sa.Integer(), nullable=False),
        sa.Column('numa_node', sa.Integer(), nullable=True),
        sa.Column('pcie_gen', sa.Integer(), nullable=True),
        sa.Column('pcie_width', sa.Integer(), nullable=True),
        sa.Column('hw_jpeg_decoder', sa.Boolean(), nullable=False),
        sa.Column('driver_version', sa.Text(), nullable=False),
        sa.Column('cuda_version', sa.Text(), nullable=False),
        sa.Column(
            'seen_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.PrimaryKeyConstraint('gpu_uuid', name=op.f('pk_gpu_devices')),
    )
    op.create_table(
        'gpu_reservations',
        sa.Column('gpu_uuid', sa.Text(), nullable=False),
        sa.Column('holder_job_id', sa.Uuid(), nullable=False),
        sa.Column(
            'reserved_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['gpu_uuid'],
            ['gpu_devices.gpu_uuid'],
            name=op.f('fk_gpu_reservations_gpu_uuid_gpu_devices'),
            ondelete='RESTRICT',
        ),
        sa.ForeignKeyConstraint(
            ['holder_job_id'],
            ['jobs.id'],
            name=op.f('fk_gpu_reservations_holder_job_id_jobs'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('gpu_uuid', name=op.f('pk_gpu_reservations')),
    )
    op.create_table(
        'search_projection_tasks',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('done_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('class_rename', 'class_merge', 'cluster_partition')",
            name=op.f('ck_search_projection_tasks_kind'),
        ),
        sa.ForeignKeyConstraint(
            ['job_id'],
            ['jobs.id'],
            name=op.f('fk_search_projection_tasks_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_search_projection_tasks_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_search_projection_tasks')),
    )
    op.create_index(
        'ix_search_projection_tasks_pending',
        'search_projection_tasks',
        ['created_at'],
        unique=False,
        postgresql_where=sa.text('done_at IS NULL'),
    )
    op.create_table(
        'projection_states',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('projection_version', sa.Integer(), nullable=False),
        sa.Column('scope', sa.Text(), nullable=False),
        sa.Column('cluster_id', sa.Integer(), nullable=True),
        sa.Column('n_points', sa.Integer(), nullable=False),
        sa.Column('n_components', sa.Integer(), nullable=False),
        sa.Column('metric', sa.Text(), nullable=False),
        sa.Column('fitted_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('reducer_storage_key', sa.Text(), nullable=True),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(
            ['job_id'],
            ['jobs.id'],
            name=op.f('fk_projection_states_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_projection_states_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('project_id', name=op.f('pk_projection_states')),
    )
