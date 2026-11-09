"""Region text: reader runs, consensus groups and their members (research plate text plan 09h 5.2).

The tables hold codes, run bookkeeping and group membership only; the readable text lives in the
OpenSearch region boxes (owner decision D-79). ``jurisdiction`` is the issuing jurisdiction of a
consensus reading (D-81), nullable with its own confidence. Every table has a non-null
``project_id``; groups and members repeat it in a composite key so a row cannot name a parent of
another project. The policies are in revision ``0005_region_text``.
"""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    Boolean,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.types import EMPTY_OBJECT, NOW, UUIDV7, check, in_list


RUN_TRIGGERS = ('region_pass', 'reread', 'consensus')
CONSENSUS_BASES = ('near_dup', 'capture_window', 'human')
CONSENSUS_METHODS = ('best_frame', 'position_vote', 'aligned_vote')
CONSENSUS_TIERS = ('confident', 'partial', 'none')


def _project_fk() -> ForeignKey:
    return ForeignKey('projects.id', ondelete='CASCADE')


class RegionTextRun(Base):
    """One reader configuration and pass. A configuration has at most one open ``region_pass``."""

    __tablename__ = 'region_text_runs'
    __table_args__ = (
        UniqueConstraint('id', 'project_id', name='uq_region_text_runs_id_project_id'),
        ForeignKeyConstraint(
            ['profile_config_id', 'profile_revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_region_text_runs_profile',
        ),
        check('trigger', in_list('trigger', RUN_TRIGGERS)),
        check('profile_pair', '(profile_config_id IS NULL) = (profile_revision IS NULL)'),
        Index('ix_region_text_runs_project', 'project_id', text('started_at DESC')),
        Index(
            'uq_region_text_runs_open_pass',
            'project_id',
            'reader',
            'reader_version',
            'profile_config_id',
            'profile_revision',
            unique=True,
            postgresql_where=text("trigger = 'region_pass' AND finished_at IS NULL"),
            postgresql_nulls_not_distinct=True,
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(_project_fk())
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    trigger: Mapped[str]
    reader: Mapped[str]
    reader_version: Mapped[str]
    profile_config_id: Mapped[uuid.UUID | None]
    profile_revision: Mapped[int | None] = mapped_column(Integer)
    params: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    counts: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    started_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    finished_at: Mapped[dt.datetime | None]
    created_by: Mapped[str]


class RegionTextConsensusGroup(Base):
    """One observation group: a burst of one vehicle read as one plate."""

    __tablename__ = 'region_text_consensus_groups'
    __table_args__ = (
        UniqueConstraint('id', 'project_id', name='uq_region_text_consensus_groups_id_project_id'),
        ForeignKeyConstraint(
            ['run_id', 'project_id'],
            ['region_text_runs.id', 'region_text_runs.project_id'],
            ondelete='CASCADE',
            name='fk_region_text_consensus_groups_run_project',
        ),
        check('basis', in_list('basis', CONSENSUS_BASES)),
        check('method', in_list('method', CONSENSUS_METHODS)),
        check('tier', in_list('tier', CONSENSUS_TIERS)),
        check('n_frames', 'n_frames >= 1'),
        check('n_agree', 'n_agree BETWEEN 0 AND n_frames'),
        check('jurisdiction_conf', 'jurisdiction_conf BETWEEN 0 AND 1'),
        Index('ix_region_text_consensus_groups_run', 'run_id', 'project_id'),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(_project_fk())
    run_id: Mapped[uuid.UUID]
    basis: Mapped[str]
    method: Mapped[str]
    n_frames: Mapped[int] = mapped_column(Integer)
    n_agree: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(REAL)
    tier: Mapped[str]
    jurisdiction: Mapped[str | None]
    jurisdiction_conf: Mapped[float | None] = mapped_column(REAL)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class RegionTextConsensusMember(Base):
    __tablename__ = 'region_text_consensus_members'
    __table_args__ = (
        # a box is in at most one current group
        UniqueConstraint(
            'project_id', 'item_id', 'box_id', name='uq_region_text_consensus_members_box'
        ),
        ForeignKeyConstraint(
            ['group_id', 'project_id'],
            ['region_text_consensus_groups.id', 'region_text_consensus_groups.project_id'],
            ondelete='CASCADE',
            name='fk_region_text_consensus_members_group_project',
        ),
    )

    group_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    item_id: Mapped[str] = mapped_column(primary_key=True)
    box_id: Mapped[str] = mapped_column(primary_key=True)
    project_id: Mapped[uuid.UUID] = mapped_column(_project_fk())
    weight: Mapped[float] = mapped_column(REAL)
    agrees: Mapped[bool] = mapped_column(Boolean)
