"""Photos and where they came from (15a section 2.1), and the capture data read from the file.

One row per stored photo per project. ``image_id`` is the content-derived wire id (32 hex digits
stored as ``uuid``, rendered back by :mod:`src.domain.wire_ids`). The revision that creates the
tables, with their policies, is ``0010_storage_record``.
"""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.models.storage_common import fk_index, project_column
from src.db.types import EMPTY_ARRAY, NOW, check, in_list


SOURCE_STATES = ('present', 'moved', 'missing')
PROVENANCE_KINDS = ('import', 'combine')
LABEL_STATES = ('labeled', 'negative', 'unlabeled')


class Photo(Base):
    """One stored photo per content per project (replaces ``image_contents`` and the images document)."""

    __tablename__ = 'photos'
    __table_args__ = (
        UniqueConstraint('project_id', 'image_id', name='uq_photos_image_id'),
        Index(
            'uq_photos_sha256',
            'project_id',
            'content_sha256',
            unique=True,
            postgresql_where=text('content_sha256 IS NOT NULL'),
        ),
        fk_index('photos', 'ingest_job_id', partial=True),
        check('imohash', 'length(imohash) = 16'),
        check('content_sha256', 'content_sha256 IS NULL OR length(content_sha256) = 32'),
        check('size', 'size_bytes >= 0'),
        check('width', 'width > 0'),
        check('height', 'height > 0'),
        check('frame', 'frame BETWEEN 1 AND 8'),
        check('source_state', in_list('source_state', SOURCE_STATES)),
    )

    photo_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    image_id: Mapped[uuid.UUID]
    imohash: Mapped[bytes] = mapped_column(LargeBinary)
    content_sha256: Mapped[bytes | None] = mapped_column(LargeBinary)
    storage_key: Mapped[str]
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    frame: Mapped[int] = mapped_column(SmallInteger)
    source_state: Mapped[str] = mapped_column(server_default='present')
    decode_route: Mapped[str]
    source: Mapped[str]
    source_identifier: Mapped[str | None]
    ingest_run_tag: Mapped[str | None]
    ingest_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('jobs.id', ondelete='SET NULL')
    )
    ingest_request_id: Mapped[str | None]
    client_metadata: Mapped[Any | None] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class PhotoProvenance(Base):
    """Import and combine provenance: one row per photo and source."""

    __tablename__ = 'photo_provenance'
    __table_args__ = (
        Index(
            'uq_photo_provenance_source',
            'photo_key',
            'kind',
            'import_id',
            'origin_project_slug',
            'origin_image_id',
            unique=True,
            postgresql_nulls_not_distinct=True,
        ),
        fk_index('photo_provenance', 'project_id'),
        fk_index('photo_provenance', 'import_id', partial=True),
        fk_index('photo_provenance', 'combine_job_id', partial=True),
        check('kind', in_list('kind', PROVENANCE_KINDS)),
        check('label_state', 'label_state IS NULL OR ' + in_list('label_state', LABEL_STATES)),
        check('import_kind', "(kind = 'import') = (import_id IS NOT NULL)"),
    )

    provenance_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    photo_key: Mapped[int] = mapped_column(ForeignKey('photos.photo_key', ondelete='CASCADE'))
    project_id: Mapped[uuid.UUID] = project_column()
    kind: Mapped[str]
    import_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('dataset_imports.id', ondelete='CASCADE')
    )
    combine_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('combine_runs.job_id', ondelete='SET NULL')
    )
    dataset_split: Mapped[str | None]
    source_stem: Mapped[str | None]
    stratum: Mapped[str | None]
    hard_negative: Mapped[bool] = mapped_column(server_default=text('false'))
    label_state: Mapped[str | None]
    negative_for: Mapped[list[str]] = mapped_column(ARRAY(Text()), server_default=EMPTY_ARRAY)
    origin_project_slug: Mapped[str | None]
    origin_image_id: Mapped[uuid.UUID | None]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class PhotoCapture(Base):
    """Capture data read from the file; the position only under the location policy."""

    __tablename__ = 'photo_capture'
    __table_args__ = (
        fk_index('photo_capture', 'project_id'),
        check('position_pair', '(lat IS NULL) = (lon IS NULL)'),
    )

    photo_key: Mapped[int] = mapped_column(
        ForeignKey('photos.photo_key', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = project_column()
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    taken_local: Mapped[dt.datetime | None] = mapped_column(DateTime())
    taken_utc: Mapped[dt.datetime | None] = mapped_column(DateTime())
    offset_minutes: Mapped[int | None] = mapped_column(SmallInteger)
    time_source: Mapped[str | None]
    offset_source: Mapped[str | None]
    time_precision: Mapped[str | None]
    camera_make: Mapped[str | None]
    camera_model: Mapped[str | None]
    lens_model: Mapped[str | None]
    focal_length_mm: Mapped[float | None] = mapped_column(REAL)
    focal_length_35mm: Mapped[float | None] = mapped_column(REAL)
    orientation: Mapped[int | None] = mapped_column(SmallInteger)
    exif_status: Mapped[str]
    has_location: Mapped[bool] = mapped_column(server_default=text('false'))
    lat: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    lon: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    altitude_m: Mapped[float | None] = mapped_column(REAL)
    location_accuracy_m: Mapped[float | None] = mapped_column(REAL)
    heading_deg: Mapped[float | None] = mapped_column(REAL)
    location_precision: Mapped[str | None]
    burst_id: Mapped[str | None]
