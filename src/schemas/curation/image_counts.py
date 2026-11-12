"""Wire model of the per-image count backfill start."""

from __future__ import annotations

from pydantic import BaseModel


class ImageCountsBackfillStarted(BaseModel):
    job_id: str
    kind: str
