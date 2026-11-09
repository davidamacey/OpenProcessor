"""Exact content keys, near-duplicate runs, groups, members, edges, decisions and pending rows.

Research near-duplicates plan (09d section 5.1) with the changes of 02d section 3: groups and
members carry a denormalized ``project_id`` copied from their run by a trigger (so their policy is
the direct form, not an ``EXISTS`` per row), and the profile reference is the composite key with
NO ACTION. A NULL ``project_id`` is the core global index. The policies and the trigger are in
revision ``0003_near_duplicates``.
"""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.types import NOW, UUIDV7, check, in_list
from src.domain import near_dup as vocab


def _project_fk(*, name: str | None = None) -> ForeignKey:
    return ForeignKey('projects.id', ondelete='CASCADE', name=name)


class ImageContent(Base):
    """The exact-duplicate key of one stored image: the sampled ``imohash`` always, the SHA-256
    only once a second file with that ``imohash`` made it necessary (a confirmation, not the key).

    ``image_path`` is where the stored bytes are read from when that confirmation needs them.
    """

    __tablename__ = 'image_contents'
    __table_args__ = (
        UniqueConstraint(
            'project_id',
            'image_id',
            name='uq_image_contents_image',
            postgresql_nulls_not_distinct=True,
        ),
        Index('ix_image_contents_imohash', 'project_id', 'imohash'),
        Index(
            'uq_image_contents_sha256',
            'project_id',
            'content_sha256',
            unique=True,
            postgresql_where=text('content_sha256 IS NOT NULL'),
            postgresql_nulls_not_distinct=True,
        ),
        check('imohash', "imohash ~ '^[0-9a-f]{32}$'"),
        check('sha256', "content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'"),
        check('size', 'size_bytes >= 0'),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID | None] = mapped_column(_project_fk())
    imohash: Mapped[str]
    content_sha256: Mapped[str | None]
    image_id: Mapped[str]
    image_path: Mapped[str]
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class NearDupRun(Base):
    __tablename__ = 'near_dup_runs'
    __table_args__ = (
        UniqueConstraint('id', 'project_id', name='uq_near_dup_runs_id_project_id'),
        ForeignKeyConstraint(
            ['profile_config_id', 'profile_revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_near_dup_runs_profile',
        ),
        check('level', in_list('level', vocab.LEVELS)),
        check('mode', in_list('mode', vocab.RUN_MODES)),
        check('status', in_list('status', vocab.RUN_STATUSES)),
        check('core_has_no_project', "(level = 'core') = (project_id IS NULL)"),
        check('core_has_no_profile', "(level = 'core') = (profile_config_id IS NULL)"),
        check('profile_pair', '(profile_config_id IS NULL) = (profile_revision IS NULL)'),
        check('building_unfinished', "(status = 'building') = (finished_at IS NULL)"),
        Index(
            'uq_near_dup_runs_current',
            'project_id',
            'level',
            unique=True,
            postgresql_where=text("status = 'current'"),
            postgresql_nulls_not_distinct=True,
        ),
        Index(
            'uq_near_dup_runs_building',
            'project_id',
            'level',
            unique=True,
            postgresql_where=text("status = 'building'"),
            postgresql_nulls_not_distinct=True,
        ),
        Index('ix_near_dup_runs_recent', 'project_id', 'level', text('started_at DESC')),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID | None] = mapped_column(_project_fk())
    level: Mapped[str]
    mode: Mapped[str]
    status: Mapped[str]
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('near_dup_runs.id', ondelete='SET NULL', name='fk_near_dup_runs_parent_run')
    )
    profile_config_id: Mapped[uuid.UUID | None]
    profile_revision: Mapped[int | None] = mapped_column(Integer)
    params: Mapped[Any] = mapped_column(JSONB)
    space: Mapped[str]
    algorithm_version: Mapped[str]
    members_scored: Mapped[int] = mapped_column(Integer, server_default='0')
    dedup_groups: Mapped[int] = mapped_column(Integer, server_default='0')
    leak_groups: Mapped[int] = mapped_column(Integer, server_default='0')
    grouped_members: Mapped[int] = mapped_column(Integer, server_default='0')
    largest_group: Mapped[int] = mapped_column(Integer, server_default='0')
    oversized_split: Mapped[int] = mapped_column(Integer, server_default='0')
    holdout_absorbed: Mapped[int] = mapped_column(Integer, server_default='0')
    started_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    finished_at: Mapped[dt.datetime | None]


class NearDupGroup(Base):
    __tablename__ = 'near_dup_groups'
    __table_args__ = (
        UniqueConstraint('id', 'project_id', name='uq_near_dup_groups_id_project_id'),
        ForeignKeyConstraint(
            ['run_id', 'project_id'],
            ['near_dup_runs.id', 'near_dup_runs.project_id'],
            ondelete='CASCADE',
            name='fk_near_dup_groups_run_project',
        ),
        check('purpose', in_list('purpose', vocab.PURPOSES)),
        check('size', 'member_count >= 2'),
        Index('ix_near_dup_groups_run', 'run_id', 'purpose', text('member_count DESC')),
        Index('ix_near_dup_groups_lineage', 'lineage_id'),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('near_dup_runs.id', ondelete='CASCADE'))
    project_id: Mapped[uuid.UUID | None]
    purpose: Mapped[str]
    lineage_id: Mapped[uuid.UUID]
    representative_id: Mapped[str]
    member_count: Mapped[int] = mapped_column(Integer)
    min_edge_cosine: Mapped[float] = mapped_column(REAL)
    min_cosine_to_rep: Mapped[float] = mapped_column(REAL)
    has_holdout: Mapped[bool] = mapped_column(Boolean, server_default='false')


class NearDupMember(Base):
    __tablename__ = 'near_dup_members'
    __table_args__ = (
        ForeignKeyConstraint(
            ['run_id', 'project_id'],
            ['near_dup_runs.id', 'near_dup_runs.project_id'],
            ondelete='CASCADE',
            name='fk_near_dup_members_run_project',
        ),
        check('purpose', in_list('purpose', vocab.PURPOSES)),
        Index('ix_near_dup_members_group', 'group_id'),
        Index(
            'uq_near_dup_members_rep',
            'group_id',
            unique=True,
            postgresql_where=text('is_representative'),
        ),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('near_dup_runs.id', ondelete='CASCADE'), primary_key=True
    )
    purpose: Mapped[str] = mapped_column(primary_key=True)
    member_id: Mapped[str] = mapped_column(primary_key=True)
    project_id: Mapped[uuid.UUID | None]
    group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('near_dup_groups.id', ondelete='CASCADE')
    )
    is_representative: Mapped[bool] = mapped_column(Boolean)
    cosine_to_rep: Mapped[float] = mapped_column(REAL)


class NearDupEdge(Base):
    __tablename__ = 'near_dup_edges'
    __table_args__ = (
        UniqueConstraint(
            'project_id',
            'level',
            'space',
            'kind',
            'a_id',
            'b_id',
            name='uq_near_dup_edges_pair',
            postgresql_nulls_not_distinct=True,
        ),
        check('level', in_list('level', vocab.LEVELS)),
        check('kind', in_list('kind', vocab.EDGE_KINDS)),
        check('ordered', 'a_id < b_id'),
        Index('ix_near_dup_edges_b', 'project_id', 'level', 'space', 'b_id'),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID | None] = mapped_column(_project_fk())
    level: Mapped[str]
    space: Mapped[str]
    a_id: Mapped[str]
    b_id: Mapped[str]
    kind: Mapped[str]
    score: Mapped[float] = mapped_column(REAL)
    computed_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class NearDupDecision(Base):
    __tablename__ = 'near_dup_decisions'
    __table_args__ = (
        UniqueConstraint('project_id', 'level', 'a_id', 'b_id', name='uq_near_dup_decisions_pair'),
        check('level', in_list('level', vocab.PROJECT_LEVELS)),
        check('decision', in_list('decision', vocab.DECISIONS)),
        check('ordered', 'a_id < b_id'),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(_project_fk())
    level: Mapped[str]
    a_id: Mapped[str]
    b_id: Mapped[str]
    decision: Mapped[str]
    decided_by: Mapped[str]
    decided_by_user_id: Mapped[uuid.UUID | None]
    decided_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class NearDupPending(Base):
    __tablename__ = 'near_dup_pending'
    __table_args__ = (
        UniqueConstraint(
            'project_id',
            'level',
            'member_id',
            name='uq_near_dup_pending_member',
            postgresql_nulls_not_distinct=True,
        ),
        check('level', in_list('level', vocab.LEVELS)),
        check('reason', in_list('reason', vocab.PENDING_REASONS)),
        Index('ix_near_dup_pending_age', 'project_id', 'level', 'enqueued_at'),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID | None] = mapped_column(_project_fk())
    level: Mapped[str]
    member_id: Mapped[str]
    reason: Mapped[str]
    enqueued_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    claimed_by_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('jobs.id', ondelete='SET NULL')
    )


class HoldoutNearDupMove(Base):
    __tablename__ = 'holdout_near_dup_moves'
    __table_args__ = (
        UniqueConstraint('project_id', 'item_id', name='uq_holdout_near_dup_moves_item'),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = mapped_column(_project_fk())
    item_id: Mapped[str]
    anchor_item_id: Mapped[str]
    group_id: Mapped[uuid.UUID | None]
    freeze_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('holdout_freezes.id', ondelete='CASCADE')
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('near_dup_runs.id', ondelete='SET NULL')
    )
    moved_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
