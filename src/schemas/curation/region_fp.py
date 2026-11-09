"""Wire model of the false-positive centroid build start."""

from __future__ import annotations

from pydantic import BaseModel


class RegionFpCentroidsStarted(BaseModel):
    job_id: str
    kind: str
