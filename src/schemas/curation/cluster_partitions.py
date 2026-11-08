"""Wire models of the partition routes (research clustering plan 5.5, S-core)."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class ClusterRunRequest(BaseModel):
    target: Literal['item', 'region_box']
    level: Literal['class'] = 'class'


class ClusterRunAccepted(BaseModel):
    job_id: str


class ClusterPartitionView(BaseModel):
    id: uuid.UUID
    target: str
    level: str
    embedding_field: str
    method: str
    status: Literal['building', 'ready', 'failed', 'discarded']
    n_items: int | None
    n_clusters: int | None
    n_review: int | None
    n_unassigned: int | None
    metrics: dict[str, Any]
    diff: dict[str, Any] | None
    created_at: str
    parent_partition_id: uuid.UUID | None
    serving: bool


class ClusterPartitionList(BaseModel):
    partitions: list[ClusterPartitionView]


class ClusterActivationRequest(BaseModel):
    target: Literal['item', 'region_box']
    level: Literal['class'] = 'class'
    partition_id: uuid.UUID
    expected_serving_partition_id: uuid.UUID | None = Field(
        default=None,
        description='The serving partition the caller saw; null for a first activation.',
    )
    expected_revision: int | None = Field(
        default=None,
        description='The activation revision the caller saw; null for a first activation.',
    )


class ClusterActivationAccepted(BaseModel):
    job_id: str
    revision: int
    pending_partition_id: uuid.UUID
