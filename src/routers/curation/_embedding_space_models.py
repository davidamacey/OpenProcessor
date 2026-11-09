"""Wire models of the embedding space routes."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.domain.embedding_spaces import Target


class EmbeddingModelEntry(BaseModel):
    model_key: str
    version: int
    modalities: list[str]
    native_dim: int
    matryoshka_dims: list[int]
    licence: str
    available: bool
    unavailable_reason: str | None = None


class EmbeddingModelListResponse(BaseModel):
    models: list[EmbeddingModelEntry]


class EmbeddingSpaceView(BaseModel):
    space_key: str
    target: Target
    model_key: str
    model_version: int
    dim: int
    role: Literal['primary', 'optional', 'candidate']
    status: Literal['pending', 'building', 'ready', 'retiring', 'retired', 'failed']
    ready_at: dt.datetime | None
    docs_with_vector: int | None = Field(
        description='Documents holding a vector in this space; null when the index was not asked.'
    )


class EmbeddingSpaceListResponse(BaseModel):
    spaces: list[EmbeddingSpaceView]


class EmbeddingSpaceRegisterRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model_key: str
    model_version: int = Field(ge=1)
    target: Target
    dim: int | None = Field(default=None, ge=1)


class EmbeddingBuildRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    mode: Literal['build', 'backfill'] = 'backfill'
    batch_size: int = Field(default=64, ge=1, le=512)


class EmbeddingPromoteRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    force: bool = Field(default=False, description='Waive the evaluation requirement (owner call).')


class EmbeddingSpaceStartResponse(BaseModel):
    """202 of register, build and retire: the space and the job that works on it."""

    space_key: str
    target: Target
    job_id: str | None = Field(description='Null when nothing needed a job (a never-built retire).')
    run_id: str | None
    bytes_estimate: int | None


class EmbeddingPromoteResponse(BaseModel):
    primary: EmbeddingSpaceView
    previous: EmbeddingSpaceView | None
    config_epoch: int


class EmbeddingReindexRunResponse(BaseModel):
    id: str
    space_key: str
    mode: str
    scope: dict[str, Any]
    docs_total: int | None
    docs_done: int
    docs_failed: int
    docs_skipped: int
    bytes_estimate: int | None
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    job_id: str
    job_status: str | None
    job_progress: dict[str, Any]


class EmbeddingRunCancelResponse(BaseModel):
    run_id: str
    job_status: str
    abandoned: bool
    space_status: str
    clear_job_id: str | None = None
