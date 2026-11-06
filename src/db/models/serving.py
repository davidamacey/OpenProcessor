"""Which model each Triton server reports as ready (deployment data, no row-level security)."""

import datetime as dt

from sqlalchemy import Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.types import NOW, check, in_list


SERVING_STATES = ('ready', 'unavailable', 'loading', 'unloading')


class ServingReport(Base):
    __tablename__ = 'serving_reports'
    __table_args__ = (check('state', in_list('state', SERVING_STATES)),)

    server: Mapped[str] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    state: Mapped[str]
    instances: Mapped[int | None] = mapped_column(Integer)
    device_mib: Mapped[int | None] = mapped_column(Integer)
    seen_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
