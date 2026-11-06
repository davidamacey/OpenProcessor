"""Embedding model catalog, per-project embedding spaces, reindex runs and evaluation runs."""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.types import EMPTY_ARRAY, EMPTY_OBJECT, NOW, UUIDV7, check, in_list


DISTANCES = ('cosine',)
ENCODER_PRECISIONS = ('fp32', 'bf16', 'fp16', 'int8')
MODEL_VERSION_STATUSES = ('available', 'deprecated', 'retired')
SPACE_TARGETS = ('item', 'box', 'frame', 'item_text')
SPACE_ROLES = ('primary', 'optional', 'candidate')
SPACE_STATUSES = ('pending', 'building', 'ready', 'retiring', 'retired', 'failed')
SPACE_INDEX_ROLES = ('images', 'items')
REINDEX_MODES = ('build', 'backfill', 'clear')
EVALUATION_PROTOCOLS = ('label_consistency_v1', 'query_set_v1')
EVALUATION_VERDICTS = ('better', 'tie', 'worse', 'inconclusive')


class EmbeddingModelVersion(Base):
    __tablename__ = 'embedding_model_versions'
    __table_args__ = (
        UniqueConstraint('model_key', 'version', name='uq_embedding_model_versions_key_version'),
        check('model_key', "model_key ~ '^[a-z][a-z0-9_]{1,39}$'"),
        check('version', 'version >= 1'),
        check(
            'modalities',
            "modalities <@ ARRAY['image', 'text'] AND cardinality(modalities) >= 1",
        ),
        check('native_dim', 'native_dim BETWEEN 8 AND 4096'),
        check('distance', in_list('distance', DISTANCES)),
        check('precision', in_list('precision', ENCODER_PRECISIONS)),
        check('parity', 'engine_parity_min_cos BETWEEN 0 AND 1'),
        check('status', in_list('status', MODEL_VERSION_STATUSES)),
        check('image_model', "(triton_image_model IS NOT NULL) = ('image' = ANY(modalities))"),
        check('text_model', "(triton_text_model IS NOT NULL) = ('text' = ANY(modalities))"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    model_key: Mapped[str]
    version: Mapped[int] = mapped_column(Integer)
    modalities: Mapped[list[str]] = mapped_column(ARRAY(Text()))
    native_dim: Mapped[int] = mapped_column(Integer)
    matryoshka_dims: Mapped[list[int]] = mapped_column(ARRAY(Integer()), server_default=EMPTY_ARRAY)
    distance: Mapped[str]
    input_spec: Mapped[Any] = mapped_column(JSONB)
    input_spec_sha256: Mapped[str]
    weights_ref: Mapped[str]
    weights_sha256: Mapped[str]
    precision: Mapped[str]
    triton_image_model: Mapped[str | None]
    triton_text_model: Mapped[str | None]
    engine_parity_min_cos: Mapped[float | None] = mapped_column(REAL)
    licence: Mapped[str]
    status: Mapped[str]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]


class EmbeddingSpace(Base):
    __tablename__ = 'embedding_spaces'
    __table_args__ = (
        UniqueConstraint(
            'project_id', 'target', 'space_key', name='uq_embedding_spaces_target_space_key'
        ),
        UniqueConstraint(
            'project_id', 'index_role', 'vector_field', name='uq_embedding_spaces_vector_field'
        ),
        check('space_key', "space_key ~ '^[a-z][a-z0-9_]{2,47}$'"),
        check('target', in_list('target', SPACE_TARGETS)),
        check('dim', 'dim BETWEEN 8 AND 4096'),
        check('role', in_list('role', SPACE_ROLES)),
        check('status', in_list('status', SPACE_STATUSES)),
        check('index_role', in_list('index_role', SPACE_INDEX_ROLES)),
        check('box_has_nested_path', "(target = 'box') = (nested_path IS NOT NULL)"),
        check('frame_in_images', "(target = 'frame') = (index_role = 'images')"),
        check('primary_is_ready', "role <> 'primary' OR status = 'ready'"),
        Index(
            'uq_embedding_spaces_one_primary',
            'project_id',
            'target',
            unique=True,
            postgresql_where=text("role = 'primary'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    space_key: Mapped[str]
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            'embedding_model_versions.id',
            ondelete='RESTRICT',
            name='fk_embedding_spaces_model_version',
        )
    )
    target: Mapped[str]
    dim: Mapped[int] = mapped_column(Integer)
    role: Mapped[str]
    status: Mapped[str]
    index_role: Mapped[str]
    vector_field: Mapped[str]
    state_field: Mapped[str | None]
    nested_path: Mapped[str | None]
    knn_method: Mapped[Any] = mapped_column(JSONB)
    revision: Mapped[int] = mapped_column(BigInteger, server_default='1')
    built_by_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('jobs.id', ondelete='SET NULL')
    )
    ready_at: Mapped[dt.datetime | None]
    retired_at: Mapped[dt.datetime | None]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]


class EmbeddingReindexRun(Base):
    __tablename__ = 'embedding_reindex_runs'
    __table_args__ = (check('mode', in_list('mode', REINDEX_MODES)),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    space_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('embedding_spaces.id', ondelete='CASCADE')
    )
    # No ON DELETE rule: a job that a run still references is never pruned.
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('jobs.id'), unique=True)
    mode: Mapped[str]
    scope: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    docs_total: Mapped[int | None] = mapped_column(BigInteger)
    docs_done: Mapped[int] = mapped_column(BigInteger, server_default='0')
    docs_failed: Mapped[int] = mapped_column(BigInteger, server_default='0')
    docs_skipped: Mapped[int] = mapped_column(BigInteger, server_default='0')
    bytes_estimate: Mapped[int | None] = mapped_column(BigInteger)
    started_at: Mapped[dt.datetime | None]
    finished_at: Mapped[dt.datetime | None]


class EmbeddingEvaluationRun(Base):
    __tablename__ = 'embedding_evaluation_runs'
    __table_args__ = (
        check('protocol', in_list('protocol', EVALUATION_PROTOCOLS)),
        check('verdict', in_list('verdict', EVALUATION_VERDICTS)),
        check('distinct_spaces', 'baseline_space_id <> candidate_space_id'),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    baseline_space_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            'embedding_spaces.id',
            ondelete='CASCADE',
            name='fk_embedding_evaluation_runs_baseline_space',
        )
    )
    candidate_space_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            'embedding_spaces.id',
            ondelete='CASCADE',
            name='fk_embedding_evaluation_runs_candidate_space',
        )
    )
    protocol: Mapped[str]
    sample_spec: Mapped[Any] = mapped_column(JSONB)
    metrics: Mapped[Any] = mapped_column(JSONB)
    verdict: Mapped[str]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]
