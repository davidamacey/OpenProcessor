"""Wire shape of ``GET {P}/training_cohorts``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class TrainingCohort(BaseModel):
    id: str
    label: str
    description: str
    cutoffs: dict[str, float]
    endpoint: str
    params: dict[str, Any]
    row_kind: Literal['crop', 'region']


class TrainingCohortList(BaseModel):
    cohorts: list[TrainingCohort]
