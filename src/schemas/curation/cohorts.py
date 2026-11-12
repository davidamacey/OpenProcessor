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


class CohortColumn(BaseModel):
    """One cohort of the summary: how to fetch its rows and how many it holds in all."""

    id: str
    label: str
    description: str
    cutoffs: dict[str, float]
    endpoint: str
    params: dict[str, Any]
    row_kind: Literal['crop', 'region']
    total: int = Field(description='Items the cohort holds, over every class and none.')


class CohortRow(BaseModel):
    """One class (or detector label) with its count in every cohort."""

    class_id: int | None = Field(description='Null for a detector label the registry lacks.')
    class_name: str
    kind: Literal['class', 'detector'] = Field(
        description='`class`: an active registry class. `detector`: a label the detector named '
        'on items without a class, for which the registry has no class.'
    )
    counts: dict[str, int] = Field(description='Items per cohort id; every cohort is present.')


class TrainingCohortSummary(BaseModel):
    """``GET /training_cohorts/summary``: the classes x cohorts matrix, counted in one request."""

    cohorts: list[CohortColumn]
    rows: list[CohortRow] = Field(description='Classes by id, then the detector-only labels.')
