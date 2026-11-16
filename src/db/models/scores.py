"""Per-item machine scores: probe predictions, scores, projection coordinates, assist answers (15a 2.3)."""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.models.storage_common import fk_index, project_column
from src.db.types import EMPTY_JSON_ARRAY, NOW, check, in_list


SCORE_KINDS = ('uniqueness', 'mistakenness')


def _item_key() -> Mapped[int]:
    return mapped_column(
        BigInteger, ForeignKey('items.item_key', ondelete='CASCADE'), primary_key=True
    )


class ProbePrediction(Base):
    """The probe's class for an item; ``confidence`` feeds the wire ``probe_actionable``."""

    __tablename__ = 'probe_predictions'
    __table_args__ = (fk_index('probe_predictions', 'project_id'),)

    item_key: Mapped[int] = _item_key()
    project_id: Mapped[uuid.UUID] = project_column()
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    class_name: Mapped[str | None]
    class_id_hint: Mapped[int | None] = mapped_column(Integer)
    entropy: Mapped[float | None] = mapped_column(REAL)
    confidence: Mapped[float | None] = mapped_column(REAL)
    margin: Mapped[float | None] = mapped_column(REAL)
    scored_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class ItemScore(Base):
    __tablename__ = 'item_scores'
    __table_args__ = (
        fk_index('item_scores', 'project_id'),
        check('kind', in_list('kind', SCORE_KINDS)),
    )

    item_key: Mapped[int] = _item_key()
    project_id: Mapped[uuid.UUID] = project_column()
    kind: Mapped[str] = mapped_column(primary_key=True)
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    score: Mapped[float] = mapped_column(REAL)
    scored_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class ItemProjection(Base):
    __tablename__ = 'item_projections'
    __table_args__ = (fk_index('item_projections', 'project_id'),)

    item_key: Mapped[int] = _item_key()
    project_id: Mapped[uuid.UUID] = project_column()
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    x: Mapped[float] = mapped_column(REAL)
    y: Mapped[float] = mapped_column(REAL)


class AssistAnswer(Base):
    __tablename__ = 'assist_answers'
    __table_args__ = (fk_index('assist_answers', 'project_id'),)

    item_key: Mapped[int] = _item_key()
    project_id: Mapped[uuid.UUID] = project_column()
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    run_id: Mapped[uuid.UUID]
    run_created_at: Mapped[dt.datetime]
    class_name: Mapped[str | None]
    class_confidence: Mapped[str | None]
    attributes: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_JSON_ARRAY)
    box_attributes: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_JSON_ARRAY)
