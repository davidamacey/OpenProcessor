"""The current state of an item and of its boxes: the fold's output (15a section 2.4).

Both tables are derived (R7): :func:`src.db.repos.state.upsert_item_states` writes only rows that
differ, and a rebuild refolds them in chunks. ``items`` is never emptied.
"""

import datetime as dt
import uuid

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Identity,
    Index,
    Integer,
    SmallInteger,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.models.storage_common import RealArray, fk_index, project_column
from src.db.types import check, in_list


GEOMETRY_SOURCES = ('detector', 'human', 'import')
AUTOVACUUM = {
    'autovacuum_vacuum_scale_factor': 0.01,
    'autovacuum_vacuum_threshold': 1000,
    'autovacuum_vacuum_cost_limit': 2000,
}


class ItemStateRow(Base):
    """One narrow row per item; the class name is joined from the registry, never copied."""

    __tablename__ = 'item_state'
    __table_args__ = (
        Index(
            'ix_item_state_class_op',
            'class_op_id',
            postgresql_where=text('class_op_id IS NOT NULL'),
        ),
        fk_index('item_state', 'project_id'),
        check('geometry_source', in_list('geometry_source', GEOMETRY_SOURCES)),
        {'postgresql_with': AUTOVACUUM},
    )

    item_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = project_column()
    photo_key: Mapped[int] = mapped_column(BigInteger)
    crop_id: Mapped[uuid.UUID]
    bbox: Mapped[list[float]] = mapped_column(RealArray)
    area: Mapped[float] = mapped_column(REAL)
    rank: Mapped[int] = mapped_column(SmallInteger)
    geometry_source: Mapped[str]
    geometry_revision: Mapped[int] = mapped_column(Integer)
    class_id: Mapped[int | None] = mapped_column(Integer)
    class_source: Mapped[str | None]
    label_source: Mapped[str | None]
    class_validated: Mapped[bool] = mapped_column(server_default=text('false'))
    class_unresolved: Mapped[bool] = mapped_column(server_default=text('false'))
    class_labeled_at: Mapped[dt.datetime | None]
    class_confidence: Mapped[float | None] = mapped_column(REAL)
    class_claim_id: Mapped[int | None] = mapped_column(BigInteger)
    class_op_id: Mapped[uuid.UUID | None]
    class_opinion_source: Mapped[str | None]
    vlm_suggestion_dismissed: Mapped[bool] = mapped_column(server_default=text('false'))
    needs_new_class: Mapped[bool] = mapped_column(server_default=text('false'))
    probe_dismissed_class: Mapped[str | None]
    excluded: Mapped[bool] = mapped_column(server_default=text('false'))
    excluded_reason: Mapped[str | None]
    excluded_at: Mapped[dt.datetime | None]
    holdout: Mapped[bool] = mapped_column(server_default=text('false'))
    review_dismissed_at: Mapped[dt.datetime | None]
    audit_sample: Mapped[bool] = mapped_column(server_default=text('false'))
    audit_outcome: Mapped[str | None]
    combine_conflict: Mapped[bool] = mapped_column(server_default=text('false'))
    import_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(Uuid()))
    region_status: Mapped[str | None]
    region_validated: Mapped[bool] = mapped_column(server_default=text('false'))
    region_verifier: Mapped[str | None]
    region_label_source: Mapped[str | None]
    region_count: Mapped[int] = mapped_column(SmallInteger, server_default='0')
    region_rejected_count: Mapped[int] = mapped_column(SmallInteger, server_default='0')
    region_max_score: Mapped[float | None] = mapped_column(REAL)
    region_revision: Mapped[int] = mapped_column(Integer)
    region_set_complete: Mapped[bool | None]
    region_seen_seq: Mapped[int] = mapped_column(Integer, server_default='0')
    geometry_pinned: Mapped[bool] = mapped_column(server_default=text('false'))
    moved_partition_id: Mapped[uuid.UUID | None]
    moved_cluster_uid: Mapped[uuid.UUID | None]
    created_at: Mapped[dt.datetime]
    updated_at: Mapped[dt.datetime]


class BoxStateRow(Base):
    """One row per live box: the machine box folded with its box claims."""

    __tablename__ = 'box_state'
    __table_args__ = (
        UniqueConstraint('item_key', 'box_id', name='uq_box_state_item_box'),
        fk_index('box_state', 'project_id'),
        fk_index('box_state', 'box_key', partial=True),
        {'postgresql_with': AUTOVACUUM},
    )

    box_row_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    item_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='CASCADE')
    )
    box_id: Mapped[str]
    box_key: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey('region_boxes.box_key', ondelete='SET NULL')
    )
    bbox: Mapped[list[float]] = mapped_column(RealArray)
    state: Mapped[str]
    score: Mapped[float | None] = mapped_column(REAL)
    detector: Mapped[str | None]
    detector_version: Mapped[str | None]
    source: Mapped[str | None]
    bbox_correct: Mapped[bool | None]
    confidence: Mapped[str | None]
    rejection_reason: Mapped[str | None]
    text: Mapped[str | None]
    text_source: Mapped[str | None]
    text_choice: Mapped[str | None]
    text_disagreement: Mapped[bool | None]
    text_fold: Mapped[str | None]
    ordinal: Mapped[int] = mapped_column(SmallInteger)
    detected_at: Mapped[dt.datetime | None]
