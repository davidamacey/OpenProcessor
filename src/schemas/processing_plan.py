"""Wire models for request-driven processing plans (#67).

The vocabulary, dependencies and per-route rules are in ``src/domain/processing_plan.py``; these
models only carry a plan in and a report out. Branch names are validated by ``resolve`` so every
rejection carries one reason code, not a pydantic literal error.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.domain.processing_plan import BRANCHES, Branch, PlanRequest


class BranchCondition(BaseModel):
    model_config = ConfigDict(extra='forbid')

    if_detected: list[str] = Field(min_length=1, max_length=64)


class ProcessingPlan(BaseModel):
    """The ``plan`` a request sends: the branches to run and optional ``when`` conditions."""

    model_config = ConfigDict(extra='forbid')

    branches: list[str] = Field(max_length=len(BRANCHES) * 2)
    when: dict[str, BranchCondition] = Field(default_factory=dict)

    def to_request(self) -> PlanRequest:
        return PlanRequest(
            branches=tuple(self.branches),
            when={name: tuple(c.if_detected) for name, c in self.when.items()},
        )


class BranchOutcome(BaseModel):
    status: Literal['ok', 'error', 'condition_false', 'disabled']
    error: str | None = None
    ms: float


class PlanReport(BaseModel):
    """What a plan-aware response reports: the resolved branches and how each one went."""

    branches: list[Branch]
    outcomes: dict[Branch, BranchOutcome]
