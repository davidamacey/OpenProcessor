"""Tables of the control-plane schema, group 'base', in creation order."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


TABLES = (
    'project_slugs',
    'projects',
    'project_indexes',
    'deployment_counters',
    'deployment_settings',
    'class_registries',
    'project_classes',
    'config_documents',
    'config_revisions',
    'config_activations',
    'config_activation_events',
    'vlm_endpoint_probes',
    'config_runtime_reports',
    'project_settings',
    'project_keymaps',
    'project_state',
)


def create() -> None:
    op.create_table(
        'project_slugs',
        sa.Column('slug', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('retired_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "slug ~ '^[a-z][a-z0-9-]{1,31}$' AND slug !~ '--' AND slug !~ '-$'",
            name=op.f('ck_project_slugs_slug'),
        ),
        sa.PrimaryKeyConstraint('slug', name=op.f('pk_project_slugs')),
    )
    op.create_table(
        'projects',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('lock_key', sa.Integer(), sa.Identity(always=True), nullable=False),
        sa.Column('slug', sa.Text(), nullable=False),
        sa.Column('display_name', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), server_default='', nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('pre_delete_status', sa.Text(), nullable=True),
        sa.Column('revision', sa.BigInteger(), server_default='1', nullable=False),
        sa.Column('config_epoch', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('paused_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('paused_by', sa.Text(), nullable=True),
        sa.Column('origin', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('index_prefix', sa.Text(), nullable=False),
        sa.Column('storage_prefix', sa.Text(), nullable=False),
        sa.Column('model_prefix', sa.Text(), nullable=False),
        sa.Column('mlflow_experiment', sa.Text(), nullable=False),
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
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.Column('created_by_user_id', sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "(status = 'deleting') = (pre_delete_status IS NOT NULL)",
            name=op.f('ck_projects_deleting_has_prior_status'),
        ),
        sa.CheckConstraint(
            "NOT (slug IN ('combine', 'new', 'all', 'none', 'projects', 'global', 'settings', 'vlm', 'health'))",
            name=op.f('ck_projects_slug_reserved'),
        ),
        sa.CheckConstraint(
            "pre_delete_status IN ('active', 'archived', 'failed')",
            name=op.f('ck_projects_pre_delete_status'),
        ),
        sa.CheckConstraint(
            "status IN ('building', 'active', 'archived', 'deleting', 'failed')",
            name=op.f('ck_projects_status'),
        ),
        sa.ForeignKeyConstraint(
            ['slug'], ['project_slugs.slug'], name=op.f('fk_projects_slug_project_slugs')
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_projects')),
        sa.UniqueConstraint('lock_key', name=op.f('uq_projects_lock_key')),
        sa.UniqueConstraint('slug', name=op.f('uq_projects_slug')),
    )
    op.create_table(
        'project_indexes',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('index_name', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint("role IN ('images', 'items')", name=op.f('ck_project_indexes_role')),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_project_indexes_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('project_id', 'role', name=op.f('pk_project_indexes')),
        sa.UniqueConstraint('index_name', name=op.f('uq_project_indexes_index_name')),
    )
    op.create_table(
        'deployment_counters',
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('value', sa.BigInteger(), server_default='0', nullable=False),
        sa.PrimaryKeyConstraint('name', name=op.f('pk_deployment_counters')),
    )
    op.create_table(
        'deployment_settings',
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('value', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('revision', sa.BigInteger(), server_default='1', nullable=False),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('updated_by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "key IN ('local_vlm_desired', 'vlm_paused_for_training', 'active_serving_profile')",
            name=op.f('ck_deployment_settings_key'),
        ),
        sa.PrimaryKeyConstraint('key', name=op.f('pk_deployment_settings')),
    )
    op.create_table(
        'class_registries',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('next_class_id', sa.Integer(), server_default='0', nullable=False),
        sa.Column('revision', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('migrated_from', sa.Text(), nullable=True),
        sa.Column('migrated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "migrated_from IN ('class_registry_file', 'opensearch_registry_doc')",
            name=op.f('ck_class_registries_migrated_from'),
        ),
        sa.CheckConstraint('next_class_id >= 0', name=op.f('ck_class_registries_next_class_id')),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_class_registries_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('project_id', name=op.f('pk_class_registries')),
    )
    op.create_table(
        'project_classes',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('class_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('class_group', sa.Text(), nullable=True),
        sa.Column('deprecated', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('merged_into', sa.Integer(), nullable=True),
        sa.Column('hotkey_letter', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), server_default='', nullable=False),
        sa.Column(
            'added_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('name_key', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "hotkey_letter ~ '^[a-z0-9]$'", name=op.f('ck_project_classes_hotkey_letter')
        ),
        sa.CheckConstraint("name_key ~ '^[a-z0-9_]+$'", name=op.f('ck_project_classes_name_key')),
        sa.CheckConstraint('class_id >= 0', name=op.f('ck_project_classes_class_id')),
        sa.CheckConstraint(
            'length(name) BETWEEN 1 AND 200 AND name = btrim(name)',
            name=op.f('ck_project_classes_name'),
        ),
        sa.CheckConstraint(
            'merged_into IS NULL OR deprecated',
            name=op.f('ck_project_classes_merged_into_deprecated'),
        ),
        sa.CheckConstraint(
            'merged_into IS NULL OR merged_into <> class_id',
            name=op.f('ck_project_classes_merged_into_other'),
        ),
        sa.ForeignKeyConstraint(
            ['project_id', 'merged_into'],
            ['project_classes.project_id', 'project_classes.class_id'],
            name='fk_project_classes_project_id_merged_into_project_classes',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['class_registries.project_id'],
            name=op.f('fk_project_classes_project_id_class_registries'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('project_id', 'class_id', name=op.f('pk_project_classes')),
    )
    op.create_index(
        'ix_project_classes_name_key', 'project_classes', ['project_id', 'name_key'], unique=False
    )
    op.create_index(
        'uq_project_classes_active_name',
        'project_classes',
        ['project_id', 'name_key'],
        unique=True,
        postgresql_where=sa.text('NOT deprecated'),
    )
    op.create_index(
        'uq_project_classes_hotkey',
        'project_classes',
        ['project_id', 'hotkey_letter'],
        unique=True,
        postgresql_where=sa.text('hotkey_letter IS NOT NULL'),
    )
    op.create_table(
        'config_documents',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('current_revision', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), server_default='', nullable=False),
        sa.Column('cloned_from_id', sa.Uuid(), nullable=True),
        sa.Column('cloned_from_revision', sa.Integer(), nullable=True),
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
        sa.Column('updated_by', sa.Text(), nullable=False),
        sa.Column('updated_by_user_id', sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "(kind IN ('vlm_endpoint', 'serving_profile')) = (project_id IS NULL)",
            name=op.f('ck_config_documents_scope'),
        ),
        sa.CheckConstraint(
            "kind IN ('prompt_pack', 'region_profile', 'open_vocab_set', 'ingest_profile', 'vlm_endpoint', 'serving_profile', 'cluster_profile', 'assist_loop_preset')",
            name=op.f('ck_config_documents_kind'),
        ),
        sa.CheckConstraint(
            "name ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'", name=op.f('ck_config_documents_name')
        ),
        sa.CheckConstraint(
            'current_revision >= 1', name=op.f('ck_config_documents_current_revision')
        ),
        sa.ForeignKeyConstraint(
            ['cloned_from_id'],
            ['config_documents.id'],
            name=op.f('fk_config_documents_cloned_from_id_config_documents'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_config_documents_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_config_documents')),
        sa.UniqueConstraint('id', 'kind', name='uq_config_documents_id_kind'),
        sa.UniqueConstraint(
            'project_id',
            'kind',
            'name',
            name='uq_config_documents_project_id_kind_name',
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_table(
        'config_revisions',
        sa.Column('config_id', sa.Uuid(), nullable=False),
        sa.Column('revision', sa.Integer(), nullable=False),
        sa.Column('body', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('body_sha256', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), server_default='', nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.CheckConstraint('revision >= 1', name=op.f('ck_config_revisions_revision')),
        sa.ForeignKeyConstraint(
            ['config_id'],
            ['config_documents.id'],
            name=op.f('fk_config_revisions_config_id_config_documents'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('config_id', 'revision', name=op.f('pk_config_revisions')),
    )
    op.create_table(
        'config_activations',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('axis', sa.Text(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('config_id', sa.Uuid(), nullable=False),
        sa.Column('revision', sa.Integer(), nullable=False),
        sa.Column(
            'activated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('activated_by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "(axis, kind) IN (('prompt_pack', 'prompt_pack'), ('detection_profile', 'region_profile'), ('vlm', 'vlm_endpoint'), ('open_vocab', 'open_vocab_set'), ('ingest_profile', 'ingest_profile'), ('clustering_item', 'cluster_profile'), ('clustering_region_box', 'cluster_profile'), ('assist_loop_preset', 'assist_loop_preset'))",
            name=op.f('ck_config_activations_axis_kind'),
        ),
        sa.CheckConstraint(
            "axis IN ('prompt_pack', 'detection_profile', 'vlm', 'open_vocab', 'ingest_profile', 'clustering_item', 'clustering_region_box', 'assist_loop_preset')",
            name=op.f('ck_config_activations_axis'),
        ),
        sa.ForeignKeyConstraint(
            ['config_id', 'kind'],
            ['config_documents.id', 'config_documents.kind'],
            name='fk_config_activations_config_id_kind_config_documents',
        ),
        sa.ForeignKeyConstraint(
            ['config_id', 'revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_config_activations_config_id_revision_config_revisions',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_config_activations_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('project_id', 'axis', name='pk_config_activations'),
    )
    op.create_table(
        'config_activation_events',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('axis', sa.Text(), nullable=False),
        sa.Column('from_config_id', sa.Uuid(), nullable=True),
        sa.Column('from_revision', sa.Integer(), nullable=True),
        sa.Column('to_config_id', sa.Uuid(), nullable=True),
        sa.Column('to_revision', sa.Integer(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column(
            'at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.Column('by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "reason IN ('activate', 'rollback', 'clone', 'upgrade')",
            name=op.f('ck_config_activation_events_reason'),
        ),
        sa.CheckConstraint(
            '(to_config_id IS NULL) = (to_revision IS NULL)',
            name=op.f('ck_config_activation_events_to_pair'),
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_config_activation_events_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_config_activation_events')),
    )
    op.create_index(
        'ix_config_activation_events_axis',
        'config_activation_events',
        ['project_id', 'axis', sa.literal_column('id DESC')],
        unique=False,
    )
    op.create_table(
        'vlm_endpoint_probes',
        sa.Column('config_id', sa.Uuid(), nullable=False),
        sa.Column('revision', sa.Integer(), nullable=False),
        sa.Column(
            'probed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.Column('ok', sa.Boolean(), nullable=False),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ['config_id', 'revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_vlm_endpoint_probes_config_id_revision_config_revisions',
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['config_id'],
            ['config_documents.id'],
            name=op.f('fk_vlm_endpoint_probes_config_id_config_documents'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('config_id', 'revision', name=op.f('pk_vlm_endpoint_probes')),
    )
    op.create_table(
        'config_runtime_reports',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('process', sa.Text(), nullable=False),
        sa.Column('host', sa.Text(), nullable=False),
        sa.Column('config_epoch', sa.BigInteger(), nullable=False),
        sa.Column(
            'applied_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_config_runtime_reports_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint(
            'project_id', 'process', 'host', name=op.f('pk_config_runtime_reports')
        ),
    )
    op.create_table(
        'project_settings',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column(
            'defaults',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column('ingest_policy', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('vlm_policy', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('vlm_scope_reps', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('search', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('location_policy', sa.Text(), server_default='store', nullable=False),
        sa.Column('revision', sa.BigInteger(), server_default='1', nullable=False),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('updated_by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "location_policy IN ('drop', 'store_coarse', 'store')",
            name=op.f('ck_project_settings_location_policy'),
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_project_settings_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('project_id', name=op.f('pk_project_settings')),
    )
    op.create_table(
        'project_keymaps',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column(
            'overrides',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column('revision', sa.BigInteger(), server_default='1', nullable=False),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('updated_by', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_project_keymaps_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('project_id', name=op.f('pk_project_keymaps')),
    )
    op.create_table(
        'project_state',
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('value', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('revision', sa.BigInteger(), server_default='1', nullable=False),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "key IN ('cluster_refresh', 'region_drain', 'region_dependency_health', "
                "'vlm_external_ack', 'open_vocab_gate')"
            ),
            name=op.f('ck_project_state_key'),
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_project_state_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('project_id', 'key', name=op.f('pk_project_state')),
    )
