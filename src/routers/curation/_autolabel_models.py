"""Wire model for the auto-label job state every ``/pipeline/auto_label*`` route
returns. Documentation/OpenAPI model: routes declare it through ``responses=``
(the state is read from the job's row). Leaf module."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AutoLabelJobState(BaseModel):
    job_id: str
    status: Literal[
        'idle',
        'queued',
        'running',
        'paused',
        'cancelling',
        'completed',
        'completed_with_errors',
        'failed',
        'cancelled',
        'interrupted',
    ]
    stage: str
    processed: int
    total: int
    started_at: float = Field(description='Unix seconds; 0 before the job starts.')
    finished_at: float = Field(description='Unix seconds; 0 while it has not finished.')
    error: str | None
    error_detail: str | None
    result: dict[str, Any] = Field(
        description=(
            'Per-stage outcomes under `stages` (cluster_residuals, auto_promote, vlm, embed, ...); '
            'each stage reports its own counters.'
        )
    )
    args: dict[str, Any] = Field(description='The arguments the run was started with.')
    pipeline: str
    backend: str | None = Field(
        description="Clustering backend, 'gpu' or 'cpu'; null before it ran."
    )
    backend_detail: str | None
    free_vram_mb: int | None
    peak_vram_mb: int | None
    stage_durations: dict[str, float] = Field(description='Wall seconds per finished stage.')
    eta_seconds: float | None
    elapsed_seconds: float
    pause_reason: str | None = Field(
        default=None, description='Why the job is paused or pausing; null otherwise.'
    )
    queue_age_seconds: float | None = Field(
        description='Seconds a queued job has waited for a worker; null once it is not queued.'
    )
    queue_max_age_seconds: float = Field(
        description='A job still queued after this long fails: no worker is running.'
    )


class AutoLabelCancelResponse(AutoLabelJobState):
    cancelled: bool = Field(description='A cancellation was requested for a running job.')


class AutoLabelActionResponse(AutoLabelJobState):
    done: bool = Field(description='There was a job to pause or resume.')


__all__ = ['AutoLabelActionResponse', 'AutoLabelCancelResponse', 'AutoLabelJobState']
