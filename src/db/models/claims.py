"""The claim log, its kind catalogue and erasure jobs (15a sections 2.2, 3 and 8).

``claims`` is append only: the trigger ``claims_append_only`` of revision ``0010_storage_record``
refuses an UPDATE, and a DELETE unless the project's delete job or a running erasure job of the
project allows it. ``claim_kinds`` holds the rows of :data:`src.domain.claim_kinds.KINDS`.
"""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    SmallInteger,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.models.storage_common import RealArray, bbox_check, fk_index, project_column
from src.db.types import EMPTY_OBJECT, NOW, UUIDV7, check, in_list
from src.domain.claim_kinds import ActorKind, SubjectKind


ACTOR_KINDS = tuple(a.value for a in ActorKind)
SUBJECT_KINDS = tuple(s.value for s in SubjectKind)
ERASURE_SUBJECT_KINDS = ('photo', 'person', 'project')
ERASURE_STATES = ('running', 'done', 'failed')


class ClaimKindRow(Base):
    """The catalogue of claim kinds (a deployment table, no project)."""

    __tablename__ = 'claim_kinds'
    __table_args__ = (
        UniqueConstraint('kind', 'subject_kind', name='uq_claim_kinds_kind_subject_kind'),
        check('subject_kind', in_list('subject_kind', SUBJECT_KINDS)),
    )

    kind: Mapped[str] = mapped_column(primary_key=True)
    subject_kind: Mapped[str]
    layer: Mapped[str]
    pins_geometry: Mapped[bool]
    asserts_object: Mapped[bool]


class Claim(Base):
    __tablename__ = 'claims'
    __table_args__ = (
        ForeignKeyConstraint(
            ['kind', 'subject_kind'], ['claim_kinds.kind', 'claim_kinds.subject_kind']
        ),
        Index(
            'ix_claims_photo',
            'photo_key',
            'claim_id',
            postgresql_where=text('photo_key IS NOT NULL'),
        ),
        Index(
            'ix_claims_entity',
            'subject_entity_key',
            'claim_id',
            postgresql_where=text('subject_entity_key IS NOT NULL'),
        ),
        Index(
            'uq_claims_withdraws',
            'withdraws_claim_id',
            unique=True,
            postgresql_where=text('withdraws_claim_id IS NOT NULL'),
        ),
        Index(
            'uq_claims_op_subject',
            'op_id',
            'kind',
            'photo_key',
            'subject_crop_id',
            'subject_box_id',
            'subject_face_id',
            'subject_pair_id',
            'subject_video_key',
            'subject_entity_key',
            unique=True,
            postgresql_nulls_not_distinct=True,
            postgresql_where=text("op_id IS NOT NULL AND kind <> 'withdraw'"),
        ),
        fk_index('claims', 'project_id'),
        check('actor_kind', in_list('actor_kind', ACTOR_KINDS)),
        check('subject_bbox', bbox_check('subject_bbox')),
        check('subject_machine_bbox', bbox_check('subject_machine_bbox')),
        check('withdraw_target', "(kind = 'withdraw') = (withdraws_claim_id IS NOT NULL)"),
        check(
            'item_subject',
            "subject_kind NOT IN ('item', 'box') OR (photo_key IS NOT NULL AND "
            'subject_crop_id IS NOT NULL AND subject_bbox IS NOT NULL AND subject_frame IS NOT NULL)',
        ),
        check('box_subject', "subject_kind <> 'box' OR subject_box_id IS NOT NULL"),
        check(
            'media_subject',
            "subject_kind <> 'media' OR (photo_key IS NULL) <> (subject_video_key IS NULL)",
        ),
    )

    claim_id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    actor_kind: Mapped[str]
    actor: Mapped[str]
    actor_user_id: Mapped[uuid.UUID | None]
    op_id: Mapped[uuid.UUID | None]
    kind: Mapped[str]
    subject_kind: Mapped[str]
    photo_key: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey('photos.photo_key', ondelete='CASCADE')
    )
    subject_crop_id: Mapped[uuid.UUID | None]
    subject_box_id: Mapped[str | None]
    subject_face_id: Mapped[uuid.UUID | None]
    subject_pair_id: Mapped[uuid.UUID | None]
    subject_video_key: Mapped[int | None] = mapped_column(BigInteger)
    subject_entity_key: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey('entities.entity_key', ondelete='CASCADE')
    )
    subject_bbox: Mapped[list[float] | None] = mapped_column(RealArray)
    subject_machine_bbox: Mapped[list[float] | None] = mapped_column(RealArray)
    subject_frame: Mapped[int | None] = mapped_column(SmallInteger)
    subject_key_hint: Mapped[int | None] = mapped_column(BigInteger)
    withdraws_claim_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey('claims.claim_id', ondelete='CASCADE')
    )
    payload: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)


class ErasureJob(Base):
    """The guard of erasure: a ``running`` row of the project lets the append-only triggers delete."""

    __tablename__ = 'erasure_jobs'
    __table_args__ = (
        fk_index('erasure_jobs', 'project_id'),
        fk_index('erasure_jobs', 'job_id', partial=True),
        check('subject_kind', in_list('subject_kind', ERASURE_SUBJECT_KINDS)),
        check('state', in_list('state', ERASURE_STATES)),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = project_column()
    subject_kind: Mapped[str]
    photo_keys: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger()))
    entity_key: Mapped[int | None] = mapped_column(BigInteger)
    state: Mapped[str]
    counts: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    finished_at: Mapped[dt.datetime | None]
