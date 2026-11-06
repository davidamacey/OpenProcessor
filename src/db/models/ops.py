"""Upgrade bookkeeping."""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.types import NOW, UUIDV7, check, in_list


UPGRADE_STEPS = ('export_bundle', 'load', 'verify', 'delete_sources')
UPGRADE_STATUSES = ('running', 'completed', 'failed')


class UpgradeRun(Base):
    __tablename__ = 'upgrade_runs'
    __table_args__ = (
        check('step', in_list('step', UPGRADE_STEPS)),
        check('status', in_list('status', UPGRADE_STATUSES)),
        Index(
            'uq_upgrade_step_done',
            'from_version',
            'step',
            unique=True,
            postgresql_where=text("status = 'completed'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    from_version: Mapped[str]
    to_version: Mapped[str]
    step: Mapped[str]
    status: Mapped[str]
    source_fingerprint: Mapped[str]
    report: Mapped[Any | None] = mapped_column(JSONB)
    started_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    finished_at: Mapped[dt.datetime | None]
