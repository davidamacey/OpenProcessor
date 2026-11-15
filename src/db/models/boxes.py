"""Machine boxes of the region stage, their text readings and OCR lines (15a section 2.3)."""

import datetime as dt
import uuid

from sqlalchemy import BigInteger, ForeignKey, Identity, Integer, SmallInteger, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.models.storage_common import RealArray, bbox_check, fk_index, project_column
from src.db.types import NOW, check, in_list
from src.domain.region_boxes import BOX_STATES


READERS = ('ocr', 'vlm')


class RegionBox(Base):
    """A machine box of the region stage, or an import's proposed box."""

    __tablename__ = 'region_boxes'
    __table_args__ = (
        UniqueConstraint('item_key', 'box_id', name='uq_region_boxes_item_box'),
        fk_index('region_boxes', 'project_id'),
        fk_index('region_boxes', 'import_id', partial=True),
        check('bbox', bbox_check('bbox')),
        check('state', in_list('state', tuple(BOX_STATES))),
    )

    box_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    item_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='CASCADE')
    )
    box_id: Mapped[str]
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    region_seq: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[list[float]] = mapped_column(RealArray)
    state: Mapped[str]
    score: Mapped[float | None] = mapped_column(REAL)
    detector: Mapped[str | None]
    detector_version: Mapped[str | None]
    source: Mapped[str | None]
    confidence: Mapped[str | None]
    rejection_reason: Mapped[str | None]
    detected_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    import_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('dataset_imports.id', ondelete='CASCADE')
    )


class BoxText(Base):
    """One reading per box and reader."""

    __tablename__ = 'box_texts'
    __table_args__ = (
        fk_index('box_texts', 'project_id'),
        check('reader', in_list('reader', READERS)),
    )

    box_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('region_boxes.box_key', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = project_column()
    reader: Mapped[str] = mapped_column(primary_key=True)
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    text: Mapped[str | None]
    text_raw: Mapped[str | None]
    text_frame: Mapped[str | None]
    invalid: Mapped[str | None]
    confidence: Mapped[float | None] = mapped_column(REAL)
    conf_char_min: Mapped[float | None] = mapped_column(REAL)
    jurisdiction: Mapped[str | None]
    jurisdiction_conf: Mapped[float | None] = mapped_column(REAL)


class TextLine(Base):
    """An OCR line: whole photo (``item_key`` NULL) or item crop."""

    __tablename__ = 'text_lines'
    __table_args__ = (
        fk_index('text_lines', 'project_id'),
        fk_index('text_lines', 'photo_key'),
        fk_index('text_lines', 'item_key', partial=True),
    )

    line_key: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = project_column()
    photo_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('photos.photo_key', ondelete='CASCADE')
    )
    item_key: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='CASCADE')
    )
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    seq: Mapped[int] = mapped_column(SmallInteger)
    text: Mapped[str]
    text_raw: Mapped[str | None]
    box_norm: Mapped[list[float]] = mapped_column(RealArray)
    polygon: Mapped[list[float] | None] = mapped_column(RealArray)
    det_score: Mapped[float | None] = mapped_column(REAL)
    rec_score: Mapped[float | None] = mapped_column(REAL)
    rel_height: Mapped[float | None] = mapped_column(REAL)
    language: Mapped[str | None]
