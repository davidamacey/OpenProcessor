"""Tables of revision 0002_research_seams, group 'embeddings': the embedding model catalog, spaces, reindex runs and evaluation runs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


TABLES = (
    'embedding_model_versions',
    'embedding_spaces',
    'embedding_reindex_runs',
    'embedding_evaluation_runs',
)


def create() -> None:
    op.create_table(
        'embedding_model_versions',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('model_key', sa.Text(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('modalities', postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column('native_dim', sa.Integer(), nullable=False),
        sa.Column(
            'matryoshka_dims',
            postgresql.ARRAY(sa.Integer()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column('distance', sa.Text(), nullable=False),
        sa.Column('input_spec', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('input_spec_sha256', sa.Text(), nullable=False),
        sa.Column('weights_ref', sa.Text(), nullable=False),
        sa.Column('weights_sha256', sa.Text(), nullable=False),
        sa.Column('precision', sa.Text(), nullable=False),
        sa.Column('triton_image_model', sa.Text(), nullable=True),
        sa.Column('triton_text_model', sa.Text(), nullable=True),
        sa.Column('engine_parity_min_cos', sa.REAL(), nullable=True),
        sa.Column('licence', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "(triton_image_model IS NOT NULL) = ('image' = ANY(modalities))",
            name=op.f('ck_embedding_model_versions_image_model'),
        ),
        sa.CheckConstraint(
            "(triton_text_model IS NOT NULL) = ('text' = ANY(modalities))",
            name=op.f('ck_embedding_model_versions_text_model'),
        ),
        sa.CheckConstraint(
            "distance IN ('cosine')", name=op.f('ck_embedding_model_versions_distance')
        ),
        sa.CheckConstraint(
            "modalities <@ ARRAY['image', 'text'] AND cardinality(modalities) >= 1",
            name=op.f('ck_embedding_model_versions_modalities'),
        ),
        sa.CheckConstraint(
            "model_key ~ '^[a-z][a-z0-9_]{1,39}$'",
            name=op.f('ck_embedding_model_versions_model_key'),
        ),
        sa.CheckConstraint(
            "precision IN ('fp32', 'bf16', 'fp16', 'int8')",
            name=op.f('ck_embedding_model_versions_precision'),
        ),
        sa.CheckConstraint(
            "status IN ('available', 'deprecated', 'retired')",
            name=op.f('ck_embedding_model_versions_status'),
        ),
        sa.CheckConstraint(
            'engine_parity_min_cos BETWEEN 0 AND 1', name=op.f('ck_embedding_model_versions_parity')
        ),
        sa.CheckConstraint(
            'native_dim BETWEEN 8 AND 4096', name=op.f('ck_embedding_model_versions_native_dim')
        ),
        sa.CheckConstraint('version >= 1', name=op.f('ck_embedding_model_versions_version')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_embedding_model_versions')),
        sa.UniqueConstraint('model_key', 'version', name='uq_embedding_model_versions_key_version'),
    )
    op.create_table(
        'embedding_spaces',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('space_key', sa.Text(), nullable=False),
        sa.Column('model_version_id', sa.Uuid(), nullable=False),
        sa.Column('target', sa.Text(), nullable=False),
        sa.Column('dim', sa.Integer(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('index_role', sa.Text(), nullable=False),
        sa.Column('vector_field', sa.Text(), nullable=False),
        sa.Column('state_field', sa.Text(), nullable=True),
        sa.Column('nested_path', sa.Text(), nullable=True),
        sa.Column('knn_method', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('revision', sa.BigInteger(), server_default='1', nullable=False),
        sa.Column('built_by_job_id', sa.Uuid(), nullable=True),
        sa.Column('ready_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retired_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "(target = 'box') = (nested_path IS NOT NULL)",
            name=op.f('ck_embedding_spaces_box_has_nested_path'),
        ),
        sa.CheckConstraint(
            "(target = 'frame') = (index_role = 'images')",
            name=op.f('ck_embedding_spaces_frame_in_images'),
        ),
        sa.CheckConstraint(
            "index_role IN ('images', 'items')", name=op.f('ck_embedding_spaces_index_role')
        ),
        sa.CheckConstraint(
            "role <> 'primary' OR status = 'ready'",
            name=op.f('ck_embedding_spaces_primary_is_ready'),
        ),
        sa.CheckConstraint(
            "role IN ('primary', 'optional', 'candidate')", name=op.f('ck_embedding_spaces_role')
        ),
        sa.CheckConstraint(
            "space_key ~ '^[a-z][a-z0-9_]{2,47}$'", name=op.f('ck_embedding_spaces_space_key')
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'building', 'ready', 'retiring', 'retired', 'failed')",
            name=op.f('ck_embedding_spaces_status'),
        ),
        sa.CheckConstraint(
            "target IN ('item', 'box', 'frame', 'item_text')",
            name=op.f('ck_embedding_spaces_target'),
        ),
        sa.CheckConstraint('dim BETWEEN 8 AND 4096', name=op.f('ck_embedding_spaces_dim')),
        sa.ForeignKeyConstraint(
            ['built_by_job_id'],
            ['jobs.id'],
            name=op.f('fk_embedding_spaces_built_by_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['model_version_id'],
            ['embedding_model_versions.id'],
            name=op.f('fk_embedding_spaces_model_version'),
            ondelete='RESTRICT',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_embedding_spaces_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_embedding_spaces')),
        sa.UniqueConstraint(
            'project_id', 'index_role', 'vector_field', name='uq_embedding_spaces_vector_field'
        ),
        sa.UniqueConstraint(
            'project_id', 'target', 'space_key', name='uq_embedding_spaces_target_space_key'
        ),
    )
    op.create_index(
        'uq_embedding_spaces_one_primary',
        'embedding_spaces',
        ['project_id', 'target'],
        unique=True,
        postgresql_where=sa.text("role = 'primary'"),
    )
    op.create_table(
        'embedding_reindex_runs',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('space_id', sa.Uuid(), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=False),
        sa.Column('mode', sa.Text(), nullable=False),
        sa.Column(
            'scope',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column('docs_total', sa.BigInteger(), nullable=True),
        sa.Column('docs_done', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('docs_failed', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('docs_skipped', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('bytes_estimate', sa.BigInteger(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "mode IN ('build', 'backfill', 'clear')", name=op.f('ck_embedding_reindex_runs_mode')
        ),
        sa.ForeignKeyConstraint(
            ['job_id'], ['jobs.id'], name=op.f('fk_embedding_reindex_runs_job_id_jobs')
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_embedding_reindex_runs_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['space_id'],
            ['embedding_spaces.id'],
            name=op.f('fk_embedding_reindex_runs_space_id_embedding_spaces'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_embedding_reindex_runs')),
        sa.UniqueConstraint('job_id', name=op.f('uq_embedding_reindex_runs_job_id')),
    )
    op.create_table(
        'embedding_evaluation_runs',
        sa.Column('id', sa.Uuid(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('job_id', sa.Uuid(), nullable=True),
        sa.Column('baseline_space_id', sa.Uuid(), nullable=False),
        sa.Column('candidate_space_id', sa.Uuid(), nullable=False),
        sa.Column('protocol', sa.Text(), nullable=False),
        sa.Column('sample_spec', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('verdict', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('created_by', sa.Text(), nullable=False),
        sa.CheckConstraint(
            "protocol IN ('label_consistency_v1', 'query_set_v1')",
            name=op.f('ck_embedding_evaluation_runs_protocol'),
        ),
        sa.CheckConstraint(
            "verdict IN ('better', 'tie', 'worse', 'inconclusive')",
            name=op.f('ck_embedding_evaluation_runs_verdict'),
        ),
        sa.CheckConstraint(
            'baseline_space_id <> candidate_space_id',
            name=op.f('ck_embedding_evaluation_runs_distinct_spaces'),
        ),
        sa.ForeignKeyConstraint(
            ['baseline_space_id'],
            ['embedding_spaces.id'],
            name=op.f('fk_embedding_evaluation_runs_baseline_space'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['candidate_space_id'],
            ['embedding_spaces.id'],
            name=op.f('fk_embedding_evaluation_runs_candidate_space'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['job_id'],
            ['jobs.id'],
            name=op.f('fk_embedding_evaluation_runs_job_id_jobs'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            name=op.f('fk_embedding_evaluation_runs_project_id_projects'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_embedding_evaluation_runs')),
    )
