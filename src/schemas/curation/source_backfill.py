"""Wire models of the source backfill start."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceBackfillRequest(BaseModel):
    """Which steps to run; omitted, all of them."""

    steps: list[Literal['content', 'capture', 'location_purge']] | None = Field(
        default=None,
        description='`content`: the exact-content key; `capture`: EXIF time, camera and GPS under '
        "the project's location policy; `location_purge`: bring stored coordinates within the "
        'policy. Omitted: all three, in that order.',
        min_length=1,
    )


class SourceBackfillStarted(BaseModel):
    job_id: str
    kind: str
