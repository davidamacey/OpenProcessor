"""Tables of the control-plane schema, group 'domain', in creation order."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


TABLES = (
    'dataset_uploads',
    'dataset_imports',
    'dataset_import_images',
    'combine_runs',
    'combine_sources',
    'ingest_sources',
    'ingest_source_files',
    'label_operations',
    'label_operation_members',
    'holdout_freezes',
    'dataset_exports',
    'training_runs',
    'models',
    'model_engines',
    'promotions',
    'model_shares',
    'bakeoff_runs',
    'upgrade_runs',
)


def create() -> None:
    op.create_table(
        'dataset_uploads',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('sha256', sa.Text(), nullable=True),
        sa.Column('storage_key', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('extract_job_id', sa.Uuid(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('receiving', 'received', 'extracting', 'extracted', 'failed', 'expired')",
            name=op.f('ck_dataset_uploads_status'),
        ),
        sa.CheckConstraint('size_bytes >= 0', name=op.f('ck_dataset_uploads_size_bytes')),
        sa.ForeignKeyConstraint(
            ['extract_job_id'],
            ['jobs.id'],
            name=op.f('fk_dataset_uploads_extract_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_dataset_uploads_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_dataset_uploads')),
        sa.UniqueConstraint('storage_key', name=op.f('uq_dataset_uploads_storage_key')),
    )
    op.create_table(
        'dataset_imports',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column('undo_job_id', sa.Uuid(), nullable=True),
        sa.Column('upload_id', sa.Uuid(), nullable=True),
        sa.Column('source_key', sa.Text(), nullable=True),
        sa.Column('format', sa.Text(), nullable=False),
        sa.Column('mapping', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('options', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('scan', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('report', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('undone_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            '(upload_id IS NULL) <> (source_key IS NULL)',
            name=op.f('ck_dataset_imports_one_source'),
        ),
        sa.ForeignKeyConstraint(
            ['job_id'],
            ['jobs.id'],
            name=op.f('fk_dataset_imports_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_dataset_imports_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['undo_job_id'],
            ['jobs.id'],
            name=op.f('fk_dataset_imports_undo_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['upload_id'],
            ['dataset_uploads.id'],
            name=op.f('fk_dataset_imports_upload_id_dataset_uploads'),
            ondelete='SET NULL',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_dataset_imports')),
        sa.UniqueConstraint('job_id', name=op.f('uq_dataset_imports_job_id')),
    )
    op.create_table(
        'dataset_import_images',
        sa.Column('import_id', sa.Uuid(), nullable=False),
        sa.Column('image_key', sa.Text(), nullable=False),
        sa.Column('image_id', sa.Text(), nullable=True),
        sa.Column(
            'item_ids', postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'written', 'skipped', 'failed', 'undone')",
            name=op.f('ck_dataset_import_images_status'),
        ),
        sa.ForeignKeyConstraint(
            ['import_id'],
            ['dataset_imports.id'],
            name=op.f('fk_dataset_import_images_import_id_dataset_imports'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('import_id', 'image_key', name=op.f('pk_dataset_import_images')),
    )
    op.create_index(
        'ix_dataset_import_images_pending',
        'dataset_import_images',
        ['import_id'],
        unique=False,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_table(
        'combine_runs',
        sa.Column('job_id', sa.Uuid(), nullable=False),
        sa.Column('target_project_id', sa.Uuid(), nullable=False),
        sa.Column('holdout_mode', sa.Text(), nullable=False),
        sa.Column('class_mapping', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "holdout_mode IN ('union', 'recompute', 'drop')",
            name=op.f('ck_combine_runs_holdout_mode'),
        ),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'], name=op.f('fk_combine_runs_job_id_jobs'), ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['target_project_id'],
            ['projects.id'],
            name=op.f('fk_combine_runs_target_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('job_id', name=op.f('pk_combine_runs')),
    )
    op.create_table(
        'combine_sources',
        sa.Column('job_id', sa.Uuid(), nullable=False),
        sa.Column('position', sa.SmallInteger(), nullable=False),
        sa.Column('source_project_id', sa.Uuid(), nullable=True),
        sa.Column('source_slug', sa.Text(), nullable=False),
        sa.CheckConstraint('position BETWEEN 0 AND 7', name=op.f('ck_combine_sources_position')),
        sa.ForeignKeyConstraint(
            ['job_id'],
            ['combine_runs.job_id'],
            name=op.f('fk_combine_sources_job_id_combine_runs'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['source_project_id'],
            ['projects.id'],
            name=op.f('fk_combine_sources_source_project_id_projects'),
            ondelete='SET NULL',
        ),
        sa.PrimaryKeyConstraint('job_id', 'position', name=op.f('pk_combine_sources')),
    )
    op.create_table(
        'ingest_sources',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('schedule', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('enabled', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('revision', sa.BigInteger(), server_default='1', nullable=False),
        sa.Column('last_poll_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('scan_state', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_ingest_sources_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_ingest_sources')),
        sa.UniqueConstraint('project_id', 'name', name='uq_ingest_sources_project_id_name'),
    )
    op.create_table(
        'ingest_source_files',
        sa.Column('source_id', sa.Uuid(), nullable=False),
        sa.Column('file_key', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column('attempts', sa.Integer(), server_default='0', nullable=False),
        sa.Column('image_id', sa.Text(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column(
            'enqueued_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'claimed', 'done', 'failed', 'skipped')",
            name=op.f('ck_ingest_source_files_status'),
        ),
        sa.ForeignKeyConstraint(
            ['job_id'],
            ['jobs.id'],
            name=op.f('fk_ingest_source_files_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['source_id'],
            ['ingest_sources.id'],
            name=op.f('fk_ingest_source_files_source_id_ingest_sources'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('source_id', 'file_key', name=op.f('pk_ingest_source_files')),
    )
    op.create_index(
        'ix_ingest_source_files_pending',
        'ingest_source_files',
        ['source_id', 'enqueued_at'],
        unique=False,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_table(
        'label_operations',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('member_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.Column('undone_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('writing', 'written', 'undoing', 'undone', 'failed')",
            name=op.f('ck_label_operations_status'),
        ),
        sa.ForeignKeyConstraint(
            ['job_id'],
            ['jobs.id'],
            name=op.f('fk_label_operations_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_label_operations_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_label_operations')),
    )
    op.create_table(
        'label_operation_members',
        sa.Column('op_id', sa.Uuid(), nullable=False),
        sa.Column('item_id', sa.Text(), nullable=False),
        sa.Column('prior', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ['op_id'],
            ['label_operations.id'],
            name=op.f('fk_label_operation_members_op_id_label_operations'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('op_id', 'item_id', name=op.f('pk_label_operation_members')),
    )
    op.create_table(
        'holdout_freezes',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('is_current', sa.Boolean(), nullable=False),
        sa.Column('source', sa.Text(), nullable=False),
        sa.Column('item_count', sa.Integer(), nullable=False),
        sa.Column('seed', sa.BigInteger(), nullable=True),
        sa.Column('snapshot_storage_key', sa.Text(), nullable=True),
        sa.Column(
            'frozen_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.Column('frozen_by', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_holdout_freezes_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_holdout_freezes')),
    )
    op.create_index(
        'uq_holdout_current',
        'holdout_freezes',
        ['project_id'],
        unique=True,
        postgresql_where=sa.text('is_current'),
    )
    op.create_table(
        'dataset_exports',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('class_names', postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column('manifest_sha256', sa.Text(), nullable=False),
        sa.Column('artifact_prefix', sa.Text(), nullable=False),
        sa.Column('split_seed', sa.BigInteger(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('yolo', 'single_class')", name=op.f('ck_dataset_exports_kind')
        ),
        sa.ForeignKeyConstraint(
            ['job_id'],
            ['jobs.id'],
            name=op.f('fk_dataset_exports_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_dataset_exports_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_dataset_exports')),
    )
    op.create_table(
        'training_runs',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=False),
        sa.Column('dataset_export_id', sa.Uuid(), nullable=True),
        sa.Column('spec', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('class_names', postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column('family', sa.Text(), nullable=False),
        sa.Column('base_model', sa.Text(), nullable=False),
        sa.Column('artifact_prefix', sa.Text(), nullable=False),
        sa.Column('manifest', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('mlflow_run_id', sa.Text(), nullable=True),
        sa.Column(
            'gpu_uuids', postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ['dataset_export_id'],
            ['dataset_exports.id'],
            name=op.f('fk_training_runs_dataset_export_id_dataset_exports'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], name=op.f('fk_training_runs_job_id_jobs')),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_training_runs_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_training_runs')),
        sa.UniqueConstraint('job_id', name=op.f('uq_training_runs_job_id')),
    )
    op.create_table(
        'models',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('origin', sa.Text(), nullable=False),
        sa.Column('training_run_id', sa.Uuid(), nullable=True),
        sa.Column('family', sa.Text(), nullable=False),
        sa.Column('task', sa.Text(), nullable=False),
        sa.Column('labels', postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column('artifact_storage_key', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "origin IN ('trained', 'uploaded', 'builtin')", name=op.f('ck_models_origin')
        ),
        sa.CheckConstraint('version >= 1', name=op.f('ck_models_version')),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_models_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['training_run_id'],
            ['training_runs.id'],
            name=op.f('fk_models_training_run_id_training_runs'),
            ondelete='SET NULL',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_models')),
        sa.UniqueConstraint(
            'project_id',
            'name',
            'version',
            name='uq_models_project_id_name_version',
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_table(
        'model_engines',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('model_id', sa.Uuid(), nullable=False),
        sa.Column('triton_model_name', sa.Text(), nullable=False),
        sa.Column('precision', sa.Text(), nullable=False),
        sa.Column('input_size', sa.Integer(), nullable=False),
        sa.Column('input_contract', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('output_contract', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('build_profile', sa.Text(), server_default='default', nullable=False),
        sa.Column('build_settings', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('build_stamp', sa.Text(), nullable=False),
        sa.Column('onnx_sha256', sa.Text(), nullable=True),
        sa.Column('engine_sha256', sa.Text(), nullable=True),
        sa.Column('device_mib_measured', sa.Integer(), nullable=True),
        sa.Column('max_batch', sa.Integer(), nullable=False),
        sa.Column('opt_batch', sa.Integer(), nullable=True),
        sa.Column('cuda_graphs', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('parity_ref', sa.Text(), nullable=True),
        sa.Column('parity_report_key', sa.Text(), nullable=True),
        sa.Column('trt_version', sa.Text(), nullable=False),
        sa.Column('gpu_arch', sa.Text(), nullable=False),
        sa.Column('engine_storage_key', sa.Text(), nullable=True),
        sa.Column('build_job_id', sa.Uuid(), nullable=True),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('built_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "build_profile IN ('default', 'small')", name=op.f('ck_model_engines_build_profile')
        ),
        sa.CheckConstraint(
            "precision IN ('fp32', 'fp16', 'int8')", name=op.f('ck_model_engines_precision')
        ),
        sa.CheckConstraint(
            "status IN ('building', 'ready', 'failed')", name=op.f('ck_model_engines_status')
        ),
        sa.ForeignKeyConstraint(
            ['build_job_id'],
            ['jobs.id'],
            name=op.f('fk_model_engines_build_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['model_id'],
            ['models.id'],
            name=op.f('fk_model_engines_model_id_models'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_model_engines')),
        sa.UniqueConstraint(
            'triton_model_name',
            'trt_version',
            'gpu_arch',
            'build_profile',
            name='uq_model_engines_build_key',
        ),
    )
    op.create_table(
        'promotions',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('model_id', sa.Uuid(), nullable=False),
        sa.Column('triton_model_name', sa.Text(), nullable=False),
        sa.Column('triton_version', sa.Integer(), nullable=False),
        sa.Column('class_remap', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column(
            'promoted_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('promoted_by', sa.Text(), nullable=False),
        sa.Column('superseded_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'], name=op.f('fk_promotions_job_id_jobs'), ondelete='SET NULL'
        ),
        sa.ForeignKeyConstraint(
            ['model_id'], ['models.id'], name=op.f('fk_promotions_model_id_models')
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_promotions_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_promotions')),
    )
    op.create_index(
        'uq_promotions_live',
        'promotions',
        ['triton_model_name'],
        unique=True,
        postgresql_where=sa.text('superseded_at IS NULL'),
    )
    op.create_table(
        'model_shares',
        sa.Column('model_id', sa.Uuid(), nullable=False),
        sa.Column('consumer_project_id', sa.Uuid(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ['consumer_project_id'],
            ['projects.id'],
            name=op.f('fk_model_shares_consumer_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['model_id'],
            ['models.id'],
            name=op.f('fk_model_shares_model_id_models'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('model_id', 'consumer_project_id', name=op.f('pk_model_shares')),
    )
    op.create_table(
        'bakeoff_runs',
        sa.Column('job_id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('eval_dataset_key', sa.Text(), nullable=False),
        sa.Column('models', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('results_storage_key', sa.Text(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'], name=op.f('fk_bakeoff_runs_job_id_jobs'), ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_bakeoff_runs_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('job_id', name=op.f('pk_bakeoff_runs')),
    )
    op.create_table(
        'upgrade_runs',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('from_version', sa.Text(), nullable=False),
        sa.Column('to_version', sa.Text(), nullable=False),
        sa.Column('step', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('source_fingerprint', sa.Text(), nullable=False),
        sa.Column('report', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            'started_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')", name=op.f('ck_upgrade_runs_status')
        ),
        sa.CheckConstraint(
            "step IN ('export_bundle', 'load', 'verify', 'delete_sources')",
            name=op.f('ck_upgrade_runs_step'),
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_upgrade_runs')),
    )
    op.create_index(
        'uq_upgrade_step_done',
        'upgrade_runs',
        ['from_version', 'step'],
        unique=True,
        postgresql_where=sa.text("status = 'completed'"),
    )
