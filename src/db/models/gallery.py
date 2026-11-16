"""The gallery tables of 15a section 2.5: entities, edges and photo state.

Designed now so no later phase redesigns a table, created empty by ``0010_storage_record`` and filled
first in phase 5. Identity rows (``entities``) never change; names, hierarchy and album fields are
claims folded into ``entity_state``.
"""

import datetime as dt
import uuid

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.models.photos import SOURCE_STATES
from src.db.models.storage_common import fk_index, project_column
from src.db.types import NOW, UUIDV7, check, in_list


ENTITY_KINDS = ('person', 'place', 'event', 'trip', 'album', 'keyword', 'memory')
EDGE_SUBJECT_KINDS = ('photo', 'face', 'item')
ALBUM_SORTS = ('position', 'taken_asc', 'taken_desc')
MEMBERSHIP_SOURCES = ('machine', 'human')
PERSON_STATES = ('unassigned', 'suggested', 'assigned', 'confirmed')
FILE_ROLES = ('motion', 'edited', 'recipe', 'raw', 'sidecar')


def _photo_key() -> Mapped[int]:
    return mapped_column(
        BigInteger, ForeignKey('photos.photo_key', ondelete='CASCADE'), primary_key=True
    )


def _entity_key() -> Mapped[int]:
    return mapped_column(
        BigInteger, ForeignKey('entities.entity_key', ondelete='CASCADE'), primary_key=True
    )


class Entity(Base):
    """People, places, events, trips, albums, keywords and memories."""

    __tablename__ = 'entities'
    __table_args__ = (
        UniqueConstraint('project_id', 'entity_id', name='uq_entities_entity_id'),
        fk_index('entities', 'cluster_uid', partial=True),
        check('kind', in_list('kind', ENTITY_KINDS)),
    )

    entity_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    entity_id: Mapped[uuid.UUID] = mapped_column(server_default=UUIDV7)
    kind: Mapped[str]
    analyser_id: Mapped[int | None] = mapped_column(ForeignKey('analysers.analyser_id'))
    cluster_uid: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('cluster_identities.cluster_uid', ondelete='SET NULL')
    )
    geonames_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class EntityEdge(Base):
    """A machine link of a subject to an entity: a result."""

    __tablename__ = 'entity_edges'
    __table_args__ = (
        fk_index('entity_edges', 'project_id'),
        check('subject_kind', in_list('subject_kind', EDGE_SUBJECT_KINDS)),
    )

    entity_key: Mapped[int] = _entity_key()
    project_id: Mapped[uuid.UUID] = project_column()
    subject_kind: Mapped[str] = mapped_column(primary_key=True)
    subject_key: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    weight: Mapped[float | None] = mapped_column(REAL)


class EntityState(Base):
    __tablename__ = 'entity_state'
    __table_args__ = (
        Index(
            'uq_entity_state_keyword',
            'project_id',
            'parent_key',
            'name_key',
            unique=True,
            postgresql_nulls_not_distinct=True,
            postgresql_where=text("kind = 'keyword'"),
        ),
        fk_index('entity_state', 'project_id'),
        fk_index('entity_state', 'parent_key', partial=True),
        fk_index('entity_state', 'cover_photo_key', partial=True),
        check('sort', 'sort IS NULL OR ' + in_list('sort', ALBUM_SORTS)),
    )

    entity_key: Mapped[int] = _entity_key()
    project_id: Mapped[uuid.UUID] = project_column()
    kind: Mapped[str]
    name: Mapped[str | None]
    name_key: Mapped[str | None]
    parent_key: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey('entities.entity_key', ondelete='CASCADE')
    )
    path: Mapped[str | None]
    description: Mapped[str | None]
    cover_photo_key: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey('photos.photo_key', ondelete='SET NULL')
    )
    sort: Mapped[str | None]
    hidden: Mapped[bool] = mapped_column(server_default=text('false'))
    merged_into: Mapped[int | None] = mapped_column(BigInteger)
    state: Mapped[str]
    member_count: Mapped[int] = mapped_column(Integer, server_default='0')
    revision: Mapped[int] = mapped_column(BigInteger)
    updated_at: Mapped[dt.datetime]


class MembershipState(Base):
    __tablename__ = 'membership_state'
    __table_args__ = (
        fk_index('membership_state', 'project_id'),
        check('source', in_list('source', MEMBERSHIP_SOURCES)),
    )

    entity_key: Mapped[int] = _entity_key()
    project_id: Mapped[uuid.UUID] = project_column()
    subject_kind: Mapped[str] = mapped_column(primary_key=True)
    subject_key: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source: Mapped[str]
    position: Mapped[float | None] = mapped_column(Numeric())
    weight: Mapped[float | None] = mapped_column(REAL)


class FaceState(Base):
    __tablename__ = 'face_state'
    __table_args__ = (
        fk_index('face_state', 'project_id'),
        fk_index('face_state', 'person_entity_key', partial=True),
        check('person_state', in_list('person_state', PERSON_STATES)),
    )

    face_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('faces.face_key', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = project_column()
    person_entity_key: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey('entities.entity_key', ondelete='SET NULL')
    )
    person_state: Mapped[str]
    assign_score: Mapped[float | None] = mapped_column(REAL)
    assign_source: Mapped[str | None]


class PhotoState(Base):
    """Fold of photo and media claims over the capture data: what the timeline and map read."""

    __tablename__ = 'photo_state'
    __table_args__ = (fk_index('photo_state', 'project_id'),)

    photo_key: Mapped[int] = _photo_key()
    project_id: Mapped[uuid.UUID] = project_column()
    taken_local: Mapped[dt.datetime | None] = mapped_column(DateTime())
    taken_utc: Mapped[dt.datetime | None] = mapped_column(DateTime())
    time_source: Mapped[str | None]
    lat: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    lon: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    location_source: Mapped[str | None]
    favourite: Mapped[bool] = mapped_column(server_default=text('false'))
    pick: Mapped[str] = mapped_column(server_default='none')
    hidden: Mapped[bool] = mapped_column(server_default=text('false'))
    archived: Mapped[bool] = mapped_column(server_default=text('false'))
    trashed_at: Mapped[dt.datetime | None]
    title: Mapped[str | None]
    caption: Mapped[str | None]
    stack_pick_group: Mapped[uuid.UUID | None]
    revision: Mapped[int] = mapped_column(BigInteger)
    updated_at: Mapped[dt.datetime]


class PhotoFile(Base):
    """The other files of one photo (motion, edited, recipe, raw, sidecar)."""

    __tablename__ = 'photo_files'
    __table_args__ = (
        fk_index('photo_files', 'project_id'),
        check('role', in_list('role', FILE_ROLES)),
        check('source_state', in_list('source_state', SOURCE_STATES)),
    )

    photo_key: Mapped[int] = _photo_key()
    project_id: Mapped[uuid.UUID] = project_column()
    role: Mapped[str] = mapped_column(primary_key=True)
    storage_key: Mapped[str] = mapped_column(primary_key=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    content_sha256: Mapped[bytes | None] = mapped_column(LargeBinary)
    source_state: Mapped[str] = mapped_column(server_default='present')


class PhotoDerivative(Base):
    __tablename__ = 'photo_derivatives'
    __table_args__ = (fk_index('photo_derivatives', 'project_id'),)

    photo_key: Mapped[int] = _photo_key()
    project_id: Mapped[uuid.UUID] = project_column()
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    thumbhash: Mapped[bytes | None] = mapped_column(LargeBinary)
    aspect: Mapped[float] = mapped_column(REAL)
    grid_key: Mapped[str | None]
    at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class PhotoSignal(Base):
    __tablename__ = 'photo_signals'
    __table_args__ = (fk_index('photo_signals', 'project_id'),)

    photo_key: Mapped[int] = _photo_key()
    project_id: Mapped[uuid.UUID] = project_column()
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    sharpness: Mapped[float | None] = mapped_column(REAL)
    exposure: Mapped[float | None] = mapped_column(REAL)
    aesthetic: Mapped[float | None] = mapped_column(REAL)
    utility_class: Mapped[str | None]


class PhotoPlace(Base):
    """A result of the effective position: recomputed when a position claim changes it."""

    __tablename__ = 'photo_places'
    __table_args__ = (fk_index('photo_places', 'project_id'),)

    photo_key: Mapped[int] = _photo_key()
    project_id: Mapped[uuid.UUID] = project_column()
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    geonames_id: Mapped[int | None] = mapped_column(Integer)
    country_code: Mapped[str | None]
    admin1: Mapped[str | None]
    admin2: Mapped[str | None]
    locality: Mapped[str | None]
