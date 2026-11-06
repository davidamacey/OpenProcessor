"""Dataset exports, training runs, models, engines, promotions, shares and bake-off runs."""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.types import EMPTY_ARRAY, NOW, UUIDV7, check, in_list


EXPORT_KINDS = ('yolo', 'single_class')
MODEL_ORIGINS = ('trained', 'uploaded', 'builtin')
ENGINE_PRECISIONS = ('fp32', 'fp16', 'int8')
ENGINE_BUILD_PROFILES = ('default', 'small')
ENGINE_STATUSES = ('building', 'ready', 'failed')


class DatasetExport(Base):
    __tablename__ = 'dataset_exports'
    __table_args__ = (check('kind', in_list('kind', EXPORT_KINDS)),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    kind: Mapped[str]
    class_names: Mapped[list[str]] = mapped_column(ARRAY(Text()))
    manifest_sha256: Mapped[str]
    artifact_prefix: Mapped[str]
    split_seed: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class TrainingRun(Base):
    __tablename__ = 'training_runs'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    # No ON DELETE rule: a job that a run still references is never pruned.
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('jobs.id'), unique=True)
    dataset_export_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('dataset_exports.id', ondelete='SET NULL')
    )
    spec: Mapped[Any] = mapped_column(JSONB)
    class_names: Mapped[list[str]] = mapped_column(ARRAY(Text()))
    family: Mapped[str]
    base_model: Mapped[str]
    artifact_prefix: Mapped[str]
    manifest: Mapped[Any | None] = mapped_column(JSONB)
    mlflow_run_id: Mapped[str | None]
    gpu_uuids: Mapped[list[str]] = mapped_column(ARRAY(Text()), server_default=EMPTY_ARRAY)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    finished_at: Mapped[dt.datetime | None]


class Model(Base):
    __tablename__ = 'models'
    __table_args__ = (
        UniqueConstraint(
            'project_id',
            'name',
            'version',
            name='uq_models_project_id_name_version',
            postgresql_nulls_not_distinct=True,
        ),
        check('version', 'version >= 1'),
        check('origin', in_list('origin', MODEL_ORIGINS)),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE')
    )
    name: Mapped[str]
    version: Mapped[int] = mapped_column(Integer)
    origin: Mapped[str]
    training_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('training_runs.id', ondelete='SET NULL')
    )
    family: Mapped[str]
    task: Mapped[str]
    labels: Mapped[list[str]] = mapped_column(ARRAY(Text()))
    artifact_storage_key: Mapped[str]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]


class ModelEngine(Base):
    __tablename__ = 'model_engines'
    __table_args__ = (
        UniqueConstraint(
            'triton_model_name',
            'trt_version',
            'gpu_arch',
            'build_profile',
            name='uq_model_engines_triton_model_name_trt_version_gpu_arch_build_profile',
        ),
        check('precision', in_list('precision', ENGINE_PRECISIONS)),
        check('build_profile', in_list('build_profile', ENGINE_BUILD_PROFILES)),
        check('status', in_list('status', ENGINE_STATUSES)),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('models.id', ondelete='CASCADE'))
    triton_model_name: Mapped[str]
    precision: Mapped[str]
    input_size: Mapped[int] = mapped_column(Integer)
    input_contract: Mapped[Any] = mapped_column(JSONB)
    output_contract: Mapped[Any] = mapped_column(JSONB)
    build_profile: Mapped[str] = mapped_column(server_default='default')
    build_settings: Mapped[Any] = mapped_column(JSONB)
    build_stamp: Mapped[str]
    onnx_sha256: Mapped[str | None]
    engine_sha256: Mapped[str | None]
    device_mib_measured: Mapped[int | None] = mapped_column(Integer)
    max_batch: Mapped[int] = mapped_column(Integer)
    opt_batch: Mapped[int | None] = mapped_column(Integer)
    cuda_graphs: Mapped[bool] = mapped_column(server_default=text('false'))
    parity_ref: Mapped[str | None]
    parity_report_key: Mapped[str | None]
    trt_version: Mapped[str]
    gpu_arch: Mapped[str]
    engine_storage_key: Mapped[str | None]
    build_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('jobs.id', ondelete='SET NULL')
    )
    status: Mapped[str]
    built_at: Mapped[dt.datetime | None]


class Promotion(Base):
    __tablename__ = 'promotions'
    __table_args__ = (
        Index(
            'uq_promotions_live',
            'triton_model_name',
            unique=True,
            postgresql_where=text('superseded_at IS NULL'),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('models.id'))
    triton_model_name: Mapped[str]
    triton_version: Mapped[int] = mapped_column(Integer)
    class_remap: Mapped[Any | None] = mapped_column(JSONB)
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    promoted_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    promoted_by: Mapped[str]
    superseded_at: Mapped[dt.datetime | None]


class ModelShare(Base):
    __tablename__ = 'model_shares'

    model_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('models.id', ondelete='CASCADE'), primary_key=True
    )
    consumer_project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]


class BakeoffRun(Base):
    __tablename__ = 'bakeoff_runs'

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('jobs.id', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    eval_dataset_key: Mapped[str]
    models: Mapped[Any] = mapped_column(JSONB)
    results_storage_key: Mapped[str | None]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
