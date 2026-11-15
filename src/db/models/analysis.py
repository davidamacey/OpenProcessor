"""Analysers and the analysis ledger (15a section 2.3).

An analyser row names what computed a result (kind, name, version, configuration hash) and is
immutable: a new prompt-pack revision or model build is a new row (trigger ``analysers_immutable``).
The ledger says which analyser last ran on which photo: the work list of any re-run is "the analyser
is not the project's current one, or the state is not ``done``".
"""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Identity, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.models.storage_common import fk_index, project_column
from src.db.types import EMPTY_OBJECT, NOW, check, in_list


ANALYSER_KINDS = (
    'detector', 'open_vocab', 'region_stage', 'text_reader', 'vlm', 'classifier', 'probe',
    'score', 'projection', 'capture_parser', 'encoder', 'face_detector', 'assist',
    'raw_label_clustering', 'import',
)  # fmt: skip
LEDGER_STATES = ('pending', 'done', 'failed', 'skipped')


class Analyser(Base):
    __tablename__ = 'analysers'
    __table_args__ = (
        UniqueConstraint('kind', 'name', 'version', 'config_sha256', name='uq_analysers_identity'),
        check('kind', in_list('kind', ANALYSER_KINDS)),
    )

    analyser_id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    kind: Mapped[str]
    name: Mapped[str]
    version: Mapped[str]
    config_sha256: Mapped[str]
    config_ref: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class AnalysisLedger(Base):
    __tablename__ = 'analysis_ledger'
    __table_args__ = (
        fk_index('analysis_ledger', 'project_id'),
        check('state', in_list('state', LEDGER_STATES)),
    )

    photo_key: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('photos.photo_key', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = project_column()
    kind: Mapped[str] = mapped_column(primary_key=True)
    analyser_id: Mapped[int] = mapped_column(ForeignKey('analysers.analyser_id'))
    state: Mapped[str]
    reason: Mapped[str | None]
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
