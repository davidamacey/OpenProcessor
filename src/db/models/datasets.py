"""Dataset uploads and imports, combine runs, ingest sources, label operations and holdout freezes."""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.types import EMPTY_ARRAY, NOW, UUIDV7, check, in_list


UPLOAD_STATUSES = ('receiving', 'received', 'extracting', 'extracted', 'failed', 'expired')
IMPORT_IMAGE_STATUSES = ('pending', 'written', 'skipped', 'failed', 'undone')
HOLDOUT_MODES = ('union', 'recompute', 'drop')
INGEST_FILE_STATUSES = ('pending', 'claimed', 'done', 'failed', 'skipped')
LABEL_OPERATION_STATUSES = ('writing', 'written', 'undoing', 'undone', 'failed')


class DatasetUpload(Base):
    __tablename__ = 'dataset_uploads'
    __table_args__ = (
        check('size_bytes', 'size_bytes >= 0'),
        check('status', in_list('status', UPLOAD_STATUSES)),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    filename: Mapped[str]
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str | None]
    storage_key: Mapped[str] = mapped_column(unique=True)
    status: Mapped[str]
    extract_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('jobs.id', ondelete='SET NULL')
    )
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    expires_at: Mapped[dt.datetime]


class DatasetImport(Base):
    __tablename__ = 'dataset_imports'
    __table_args__ = (check('one_source', '(upload_id IS NULL) <> (source_key IS NULL)'),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('jobs.id', ondelete='SET NULL'), unique=True
    )
    undo_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('jobs.id', ondelete='SET NULL')
    )
    upload_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('dataset_uploads.id', ondelete='SET NULL')
    )
    source_key: Mapped[str | None]
    format: Mapped[str]
    mapping: Mapped[Any] = mapped_column(JSONB)
    options: Mapped[Any] = mapped_column(JSONB)
    scan: Mapped[Any | None] = mapped_column(JSONB)
    report: Mapped[Any | None] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    undone_at: Mapped[dt.datetime | None]


class DatasetImportImage(Base):
    __tablename__ = 'dataset_import_images'
    __table_args__ = (
        check('status', in_list('status', IMPORT_IMAGE_STATUSES)),
        Index(
            'ix_dataset_import_images_pending',
            'import_id',
            postgresql_where=text("status = 'pending'"),
        ),
    )

    import_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('dataset_imports.id', ondelete='CASCADE'), primary_key=True
    )
    image_key: Mapped[str] = mapped_column(primary_key=True)
    image_id: Mapped[str | None]
    item_ids: Mapped[list[str]] = mapped_column(ARRAY(Text()), server_default=EMPTY_ARRAY)
    status: Mapped[str]
    error: Mapped[str | None]


class CombineRun(Base):
    __tablename__ = 'combine_runs'
    __table_args__ = (check('holdout_mode', in_list('holdout_mode', HOLDOUT_MODES)),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('jobs.id', ondelete='CASCADE'), primary_key=True
    )
    target_project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE')
    )
    holdout_mode: Mapped[str]
    class_mapping: Mapped[Any] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class CombineSource(Base):
    __tablename__ = 'combine_sources'
    __table_args__ = (check('position', 'position BETWEEN 0 AND 7'),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('combine_runs.job_id', ondelete='CASCADE'), primary_key=True
    )
    position: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    source_project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('projects.id', ondelete='SET NULL')
    )
    source_slug: Mapped[str]


class IngestSource(Base):
    __tablename__ = 'ingest_sources'
    __table_args__ = (
        UniqueConstraint('project_id', 'name', name='uq_ingest_sources_project_id_name'),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    name: Mapped[str]
    kind: Mapped[str]
    config: Mapped[Any] = mapped_column(JSONB)
    schedule: Mapped[Any] = mapped_column(JSONB)
    enabled: Mapped[bool] = mapped_column(server_default=text('true'))
    revision: Mapped[int] = mapped_column(BigInteger, server_default='1')
    last_poll_at: Mapped[dt.datetime | None]
    scan_state: Mapped[Any | None] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class IngestSourceFile(Base):
    __tablename__ = 'ingest_source_files'
    __table_args__ = (
        check('status', in_list('status', INGEST_FILE_STATUSES)),
        Index(
            'ix_ingest_source_files_pending',
            'source_id',
            'enqueued_at',
            postgresql_where=text("status = 'pending'"),
        ),
    )

    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('ingest_sources.id', ondelete='CASCADE'), primary_key=True
    )
    file_key: Mapped[str] = mapped_column(primary_key=True)
    status: Mapped[str]
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    attempts: Mapped[int] = mapped_column(Integer, server_default='0')
    image_id: Mapped[str | None]
    error: Mapped[str | None]
    enqueued_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class LabelOperation(Base):
    __tablename__ = 'label_operations'
    __table_args__ = (check('status', in_list('status', LABEL_OPERATION_STATUSES)),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    kind: Mapped[str]
    status: Mapped[str]
    params: Mapped[Any] = mapped_column(JSONB)
    member_count: Mapped[int] = mapped_column(Integer, server_default='0')
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]
    undone_at: Mapped[dt.datetime | None]


class LabelOperationMember(Base):
    __tablename__ = 'label_operation_members'

    op_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('label_operations.id', ondelete='CASCADE'), primary_key=True
    )
    item_id: Mapped[str] = mapped_column(primary_key=True)
    prior: Mapped[Any] = mapped_column(JSONB)


class HoldoutFreeze(Base):
    __tablename__ = 'holdout_freezes'
    __table_args__ = (
        Index(
            'uq_holdout_current',
            'project_id',
            unique=True,
            postgresql_where=text('is_current'),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    is_current: Mapped[bool]
    source: Mapped[str]
    item_count: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    snapshot_storage_key: Mapped[str | None]
    frozen_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    frozen_by: Mapped[str]
