"""Tables of revision 0002_research_seams, group 'clustering': clustering calibrations, partitions, identities, memberships and activations."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


TABLES = (
    'cluster_calibrations',
    'cluster_partitions',
    'cluster_identities',
    'partition_clusters',
    'cluster_memberships',
    'cluster_activations',
    'cluster_activation_events',
    'cluster_backend_validations',
)


def create() -> None:
    op.create_table(
        'cluster_calibrations',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('target', sa.Text(), nullable=False),
        sa.Column('level', sa.Text(), nullable=False),
        sa.Column('embedding_field', sa.Text(), nullable=False),
        sa.Column('embedding_model', sa.Text(), nullable=False),
        sa.Column('preprocess_id', sa.Text(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('fitted_on', sa.Text(), nullable=False),
        sa.Column('body', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('dev_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.CheckConstraint("kind IN ('assign_bands')", name=op.f('ck_cluster_calibrations_kind')),
        sa.CheckConstraint(
            "level IN ('class', 'identity', 'scene')", name=op.f('ck_cluster_calibrations_level')
        ),
        sa.CheckConstraint(
            "target IN ('item', 'region_box', 'face', 'image')",
            name=op.f('ck_cluster_calibrations_target'),
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_cluster_calibrations_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cluster_calibrations')),
    )
    op.create_table(
        'cluster_partitions',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('target', sa.Text(), nullable=False),
        sa.Column('level', sa.Text(), nullable=False),
        sa.Column('embedding_field', sa.Text(), nullable=False),
        sa.Column('embedding_model', sa.Text(), nullable=False),
        sa.Column('preprocess_id', sa.Text(), nullable=False),
        sa.Column('method', sa.Text(), nullable=False),
        sa.Column('method_version', sa.Text(), nullable=False),
        sa.Column('profile_config_id', sa.Uuid(), nullable=True),
        sa.Column('profile_revision', sa.Integer(), nullable=True),
        sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('calibration_id', sa.Uuid(), nullable=True),
        sa.Column('pool_filter', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('input_digest', sa.Text(), nullable=False),
        sa.Column('seed', sa.BigInteger(), nullable=False),
        sa.Column('backend', sa.Text(), nullable=False),
        sa.Column('backend_detail', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('parent_partition_id', sa.Uuid(), nullable=True),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('n_items', sa.Integer(), nullable=True),
        sa.Column('n_clusters', sa.Integer(), nullable=True),
        sa.Column('n_review', sa.Integer(), nullable=True),
        sa.Column('n_unassigned', sa.Integer(), nullable=True),
        sa.Column(
            'metrics',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column('diff', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            'stage_durations',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column('peak_rss_mb', sa.Integer(), nullable=True),
        sa.Column('peak_vram_mb', sa.Integer(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.CheckConstraint("backend IN ('cpu', 'gpu')", name=op.f('ck_cluster_partitions_backend')),
        sa.CheckConstraint(
            "level IN ('class', 'identity', 'scene')", name=op.f('ck_cluster_partitions_level')
        ),
        sa.CheckConstraint(
            "method IN ('ivf', 'minibatch_kmeans', 'union_find', 'detector_class', 'edit')",
            name=op.f('ck_cluster_partitions_method'),
        ),
        sa.CheckConstraint(
            "status IN ('building', 'ready', 'failed', 'discarded')",
            name=op.f('ck_cluster_partitions_status'),
        ),
        sa.CheckConstraint(
            "target IN ('item', 'region_box', 'face', 'image')",
            name=op.f('ck_cluster_partitions_target'),
        ),
        sa.CheckConstraint(
            '(profile_config_id IS NULL) = (profile_revision IS NULL)',
            name=op.f('ck_cluster_partitions_profile_pair'),
        ),
        sa.ForeignKeyConstraint(
            ['calibration_id'],
            ['cluster_calibrations.id'],
            name=op.f('fk_cluster_partitions_calibration_id_cluster_calibrations'),
        ),
        sa.ForeignKeyConstraint(
            ['job_id'],
            ['jobs.id'],
            name=op.f('fk_cluster_partitions_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['parent_partition_id'],
            ['cluster_partitions.id'],
            name=op.f('fk_cluster_partitions_parent_partition'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['profile_config_id', 'profile_revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_cluster_partitions_profile',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_cluster_partitions_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cluster_partitions')),
        sa.UniqueConstraint('id', 'project_id', name='uq_cluster_partitions_id_project_id'),
    )
    op.create_index(
        'ix_cluster_partitions_recent',
        'cluster_partitions',
        ['project_id', 'target', 'level', sa.literal_column('created_at DESC')],
        unique=False,
    )
    op.create_table(
        'cluster_identities',
        sa.Column('cluster_uid', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('target', sa.Text(), nullable=False),
        sa.Column('level', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=True),
        sa.Column('name_source', sa.Text(), nullable=True),
        sa.Column('anchored_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('anchored_by', sa.Text(), nullable=True),
        sa.Column('promoted_class_id', sa.Integer(), nullable=True),
        sa.Column('first_partition_id', sa.Uuid(), nullable=True),
        sa.Column('last_partition_id', sa.Uuid(), nullable=True),
        sa.Column('revision', sa.BigInteger(), server_default='1', nullable=False),
        sa.CheckConstraint(
            "level IN ('class', 'identity')", name=op.f('ck_cluster_identities_level')
        ),
        sa.CheckConstraint(
            "name_source IN ('human', 'import', 'vlm')",
            name=op.f('ck_cluster_identities_name_source'),
        ),
        sa.CheckConstraint(
            "target IN ('item', 'region_box', 'face', 'image')",
            name=op.f('ck_cluster_identities_target'),
        ),
        sa.CheckConstraint(
            '(anchored_at IS NULL) = (anchored_by IS NULL)',
            name=op.f('ck_cluster_identities_anchored_pair'),
        ),
        sa.ForeignKeyConstraint(
            ['first_partition_id'],
            ['cluster_partitions.id'],
            name=op.f('fk_cluster_identities_first_partition'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['last_partition_id'],
            ['cluster_partitions.id'],
            name=op.f('fk_cluster_identities_last_partition'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id', 'promoted_class_id'],
            ['project_classes.project_id', 'project_classes.class_id'],
            name='fk_cluster_identities_promoted_class',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_cluster_identities_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('cluster_uid', name=op.f('pk_cluster_identities')),
    )
    op.create_table(
        'partition_clusters',
        sa.Column('partition_id', sa.Uuid(), nullable=False),
        sa.Column('cluster_uid', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('display_no', sa.Integer(), nullable=False),
        sa.Column('parent_uid', sa.Uuid(), nullable=True),
        sa.Column('transition', sa.Text(), nullable=False),
        sa.Column(
            'predecessor_uids',
            postgresql.ARRAY(sa.Uuid()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('dispersion', sa.REAL(), nullable=True),
        sa.Column(
            'prototypes',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "transition IN ('new', 'survived', 'split', 'merged', 'absorbed')",
            name=op.f('ck_partition_clusters_transition'),
        ),
        sa.CheckConstraint('size > 0', name=op.f('ck_partition_clusters_size')),
        sa.ForeignKeyConstraint(
            ['cluster_uid'],
            ['cluster_identities.cluster_uid'],
            name=op.f('fk_partition_clusters_cluster_uid_cluster_identities'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['partition_id', 'project_id'],
            ['cluster_partitions.id', 'cluster_partitions.project_id'],
            name='fk_partition_clusters_partition_project',
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['partition_id'],
            ['cluster_partitions.id'],
            name=op.f('fk_partition_clusters_partition_id_cluster_partitions'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('partition_id', 'cluster_uid', name=op.f('pk_partition_clusters')),
        sa.UniqueConstraint('partition_id', 'display_no', name='uq_partition_clusters_display_no'),
    )
    op.create_table(
        'cluster_memberships',
        sa.Column('partition_id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('item_key', sa.Text(), nullable=False),
        sa.Column('cluster_uid', sa.Uuid(), nullable=True),
        sa.Column('state', sa.Text(), nullable=False),
        sa.Column('score', sa.REAL(), nullable=True),
        sa.Column('margin', sa.REAL(), nullable=True),
        sa.CheckConstraint(
            "(state = 'unassigned') = (cluster_uid IS NULL)",
            name=op.f('ck_cluster_memberships_unassigned_iff_no_cluster'),
        ),
        sa.CheckConstraint(
            "state IN ('assigned', 'review', 'unassigned')",
            name=op.f('ck_cluster_memberships_state'),
        ),
        sa.ForeignKeyConstraint(
            ['partition_id', 'project_id'],
            ['cluster_partitions.id', 'cluster_partitions.project_id'],
            name='fk_cluster_memberships_partition_project',
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['partition_id'],
            ['cluster_partitions.id'],
            name=op.f('fk_cluster_memberships_partition_id_cluster_partitions'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('partition_id', 'item_key', name=op.f('pk_cluster_memberships')),
    )
    op.create_index(
        'ix_cluster_memberships_cluster',
        'cluster_memberships',
        ['partition_id', 'cluster_uid'],
        unique=False,
    )
    op.create_table(
        'cluster_activations',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('target', sa.Text(), nullable=False),
        sa.Column('level', sa.Text(), nullable=False),
        sa.Column('embedding_field', sa.Text(), nullable=False),
        sa.Column('serving_partition_id', sa.Uuid(), nullable=False),
        sa.Column('pending_partition_id', sa.Uuid(), nullable=True),
        sa.Column('revision', sa.BigInteger(), server_default='1', nullable=False),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('updated_by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "level IN ('class', 'identity', 'scene')", name=op.f('ck_cluster_activations_level')
        ),
        sa.CheckConstraint(
            "target IN ('item', 'region_box', 'face', 'image')",
            name=op.f('ck_cluster_activations_target'),
        ),
        sa.ForeignKeyConstraint(
            ['pending_partition_id'],
            ['cluster_partitions.id'],
            name=op.f('fk_cluster_activations_pending_partition'),
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_cluster_activations_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['serving_partition_id'],
            ['cluster_partitions.id'],
            name=op.f('fk_cluster_activations_serving_partition'),
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cluster_activations')),
        sa.UniqueConstraint(
            'project_id',
            'target',
            'level',
            'embedding_field',
            name='uq_cluster_activations_served_key',
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_table(
        'cluster_activation_events',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=True),
        sa.Column('target', sa.Text(), nullable=False),
        sa.Column('level', sa.Text(), nullable=False),
        sa.Column('embedding_field', sa.Text(), nullable=False),
        sa.Column('from_partition_id', sa.Uuid(), nullable=True),
        sa.Column('to_partition_id', sa.Uuid(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column(
            'at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.Column('by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "reason IN ('auto', 'human', 'rollback', 'edit', 'upgrade')",
            name=op.f('ck_cluster_activation_events_reason'),
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_cluster_activation_events_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cluster_activation_events')),
    )
    op.create_table(
        'cluster_backend_validations',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('method', sa.Text(), nullable=False),
        sa.Column('backend_versions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('embedding_field', sa.Text(), nullable=False),
        sa.Column('dataset_pin', sa.Text(), nullable=False),
        sa.Column('ari_vs_cpu', sa.REAL(), nullable=False),
        sa.Column('purity_delta', sa.REAL(), nullable=False),
        sa.Column('graph_recall', sa.REAL(), nullable=True),
        sa.Column('passed', sa.Boolean(), nullable=False),
        sa.Column(
            'validated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('validated_by', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cluster_backend_validations')),
    )
