"""Wire model of the source backfill start."""

from __future__ import annotations

from pydantic import BaseModel


class SourceBackfillStarted(BaseModel):
    job_id: str
    kind: str
