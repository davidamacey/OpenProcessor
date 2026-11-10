"""Wire model for the auto-label job state every ``/pipeline/auto_label*`` route
returns. Documentation/OpenAPI model: routes declare it through ``responses=``
(the state is read from the job's row). Leaf module."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
            'Per-stage outcomes under `stages` (embed_missing, vlm, ...); '
            'each stage reports its own counters.'
        )
    )
    args: dict[str, Any] = Field(description='The arguments the run was started with.')
    pipeline: str
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


class AutoLabelRunSummary(BaseModel):
    """``POST /pipeline/auto_label``: the synchronous run's per-stage counts.

    Keys after ``baseline`` appear only when the run got that far (``after`` / ``delta`` / ``final``
    after the stages, ``stages_error`` when the fetch failed); served with
    ``response_model_exclude_unset`` semantics through ``responses=``.
    """

    model_config = ConfigDict(extra='allow')

    stages: dict[str, Any]
    class_id: int | None
    cluster_uid: str | None
    prompt_pack: str | None
    prompt_pack_revision: int | None
    baseline: dict[str, Any]
    after: dict[str, Any] | None = None
    delta: dict[str, int] | None = None
    final: dict[str, int] | None = None
    unvalidated_remaining: int | None = None
    stages_error: str | None = None
