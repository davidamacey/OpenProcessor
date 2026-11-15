"""Item identity and the machine results about an item (15a section 2.3).

``items`` never changes in meaning and is never emptied; every result table below it can be emptied
and filled again by a re-run. A row with ``op_id`` or ``import_id`` set is not rebuildable (R2).
"""

import datetime as dt
import uuid
from typing import Any, get_args

from sqlalchemy import BigInteger, ForeignKey, Identity, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.models.storage_common import RealArray, bbox_check, fk_index, project_column
from src.db.types import EMPTY_ARRAY, NOW, check, in_list
from src.domain.results import OpinionOutcome, OpinionSource


OPINION_SOURCES = get_args(OpinionSource)
OPINION_OUTCOMES = get_args(OpinionOutcome)
NOTE_KINDS = ('proposal', 'class_mismatch', 'missed_label', 'unmatched_detection')
GAP_REASONS = ('not_selected', 'failed')


class Item(Base):
    __tablename__ = 'items'
    __table_args__ = (
        UniqueConstraint('project_id', 'crop_id', name='uq_items_crop_id'),
        fk_index('items', 'photo_key'),
    )

    item_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    photo_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('photos.photo_key', ondelete='CASCADE')
    )
    crop_id: Mapped[uuid.UUID]
    box_seq: Mapped[int] = mapped_column(Integer, server_default='0')
    geometry_seq: Mapped[int] = mapped_column(Integer, server_default='0')
    region_seq: Mapped[int] = mapped_column(Integer, server_default='0')
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class Detection(Base):
    __tablename__ = 'detections'
    __table_args__ = (
        UniqueConstraint('item_key', 'analyser_id', name='uq_detections_item_analyser'),
        fk_index('detections', 'project_id'),
        fk_index('detections', 'photo_key'),
        fk_index('detections', 'import_id', partial=True),
        check('bbox', bbox_check('bbox')),
    )

    detection_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    photo_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('photos.photo_key', ondelete='CASCADE')
    )
    item_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='CASCADE')
    )
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    bbox: Mapped[list[float]] = mapped_column(RealArray)
    class_name: Mapped[str | None]
    class_id: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(REAL)
    proposal_name: Mapped[str | None]
    blur_lap_var: Mapped[float | None] = mapped_column(REAL)
    blur_lap_ratio: Mapped[float | None] = mapped_column(REAL)
    blur_full_var: Mapped[float | None] = mapped_column(REAL)
    prompt: Mapped[str | None]
    mask_polygon: Mapped[list[float] | None] = mapped_column(RealArray)
    import_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('dataset_imports.id', ondelete='CASCADE')
    )
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class ClassOpinion(Base):
    """One machine class opinion per item and source."""

    __tablename__ = 'class_opinions'
    __table_args__ = (
        fk_index('class_opinions', 'project_id'),
        fk_index('class_opinions', 'import_id', partial=True),
        check('source', in_list('source', OPINION_SOURCES)),
        check('outcome', in_list('outcome', OPINION_OUTCOMES)),
        check('import_source', "(source = 'import_suggestion') = (import_id IS NOT NULL)"),
        check(
            'operation_source',
            "source NOT IN ('assist_propagated', 'adopted_detector') OR op_id IS NOT NULL",
        ),
    )

    item_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = project_column()
    source: Mapped[str] = mapped_column(primary_key=True)
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    class_name: Mapped[str | None]
    class_id_hint: Mapped[int | None] = mapped_column(Integer)
    outcome: Mapped[str]
    confidence: Mapped[float | None] = mapped_column(REAL)
    confidence_cat: Mapped[str | None]
    raw_label: Mapped[str | None]
    raw_label_conf: Mapped[float | None] = mapped_column(REAL)
    raw_class: Mapped[str | None]
    proposed_class: Mapped[str | None]
    empty_reason: Mapped[str | None]
    via: Mapped[str | None]
    op_id: Mapped[uuid.UUID | None]
    import_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('dataset_imports.id', ondelete='CASCADE')
    )
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class RawLabelCluster(Base):
    """Clustering of VLM raw label strings (today ``vlm_label_cluster_*`` on every item)."""

    __tablename__ = 'raw_label_clusters'

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    raw_label: Mapped[str] = mapped_column(primary_key=True)
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    cluster_id: Mapped[int] = mapped_column(Integer)
    cluster_name: Mapped[str]
    distance: Mapped[float | None] = mapped_column(REAL)


class ItemRegionResult(Base):
    """The region stage's item-level outcome; replaced on a re-run."""

    __tablename__ = 'item_region_results'
    __table_args__ = (fk_index('item_region_results', 'project_id'),)

    item_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = project_column()
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    status: Mapped[str]
    reason: Mapped[str | None]
    verified: Mapped[bool | None]
    verified_at: Mapped[dt.datetime | None]
    verifier: Mapped[str | None]
    verifier_version: Mapped[str | None]
    auto_confirmed: Mapped[bool | None]
    visible: Mapped[bool | None]
    skip_verify: Mapped[bool | None]
    gate_skip: Mapped[str | None]
    detector_chain: Mapped[list[str]] = mapped_column(ARRAY(Text()), server_default=EMPTY_ARRAY)
    set_complete: Mapped[bool | None]
    detected_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class ItemNote(Base):
    """Re-detect and import notes on an item; the wire ``proposal_chain`` is its ``proposal`` rows."""

    __tablename__ = 'item_notes'
    __table_args__ = (
        fk_index('item_notes', 'project_id'),
        fk_index('item_notes', 'item_key'),
        fk_index('item_notes', 'import_id', partial=True),
        check('kind', in_list('kind', NOTE_KINDS)),
    )

    note_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    item_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='CASCADE')
    )
    kind: Mapped[str]
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    import_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('dataset_imports.id', ondelete='CASCADE')
    )
    body: Mapped[Any] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class Face(Base):
    __tablename__ = 'faces'
    __table_args__ = (
        UniqueConstraint('project_id', 'face_id', name='uq_faces_face_id'),
        fk_index('faces', 'photo_key'),
        fk_index('faces', 'person_item_key', partial=True),
        check('landmarks', 'cardinality(landmarks) = 10'),
    )

    face_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    photo_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('photos.photo_key', ondelete='CASCADE')
    )
    face_id: Mapped[uuid.UUID]
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    person_item_key: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='SET NULL')
    )
    bbox: Mapped[list[float]] = mapped_column(RealArray)
    landmarks: Mapped[list[float]] = mapped_column(RealArray)
    det_score: Mapped[float] = mapped_column(REAL)
    quality: Mapped[float | None] = mapped_column(REAL)
    yaw_deg: Mapped[float | None] = mapped_column(REAL)
    eyes_open: Mapped[bool | None]
    side_px: Mapped[int | None] = mapped_column(Integer)


class VectorGap(Base):
    """Why a subject has no vector in a space: the reader's explanation, never a work list."""

    __tablename__ = 'vector_gaps'
    __table_args__ = (
        fk_index('vector_gaps', 'project_id'),
        check('reason', in_list('reason', GAP_REASONS)),
    )

    space_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('embedding_spaces.id', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = project_column()
    subject_key: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    reason: Mapped[str]
