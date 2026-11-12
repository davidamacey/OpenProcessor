"""Wire shape of ``GET {P}/training_cohorts``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrainingCohort(BaseModel):
    id: str
    label: str
    description: str
    cutoffs: dict[str, float]
    endpoint: str
    params: dict[str, Any]
    row_kind: Literal['crop', 'region']
    count: int | None = Field(
        default=None,
        description=(
            'Rows the cohort holds for the requested `class_id`, counted by the same query that '
            'serves its rows; null when no class was asked for.'
        ),
    )


class TrainingCohortList(BaseModel):
    cohorts: list[TrainingCohort]
