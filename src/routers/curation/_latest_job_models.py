"""Wire models for the status/cancel routes of the single-job features (scores, viz projection).

Both read ``latest_job.common_state`` plus their own keys. Leaf module."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LatestJobState(BaseModel):
    job_id: str = Field(description='Empty string while the feature has never run.')
    status: str = Field(description="'idle' without a job, else the job status.")
    started_at: float = Field(description='Unix seconds; 0 before the job starts.')
    finished_at: float = Field(description='Unix seconds; 0 while it has not finished.')
    error: str | None


class ScoresState(LatestJobState):
    """``GET /scores/status`` (and the 202 of ``POST /scores/compute``)."""

    scorers: list[str]
    processed: int
    total: int
    results: dict[str, Any] = Field(description='Per-scorer outcome of the finished job.')


class ScoresCancelResponse(ScoresState):
    cancelled: bool = Field(description='A cancellation was requested for an active job.')


class ScorerCoverage(BaseModel):
    field: str
    n_scored: int
    total: int
    pct: float


class ScoresCoverageResponse(BaseModel):
    coverage: dict[str, ScorerCoverage] = Field(description='Keyed by scorer id.')


class VizProjectionState(LatestJobState):
    """``GET /viz/projection/status`` (and the 202 of ``POST /viz/projection/rebuild``)."""

    scope: str
    cluster_uid: str | None
    n_pool: int
    n_written: int
    projection_version: str | None


class VizProjectionCancelResponse(VizProjectionState):
    cancelled: bool = Field(description='A cancellation was requested for an active job.')
