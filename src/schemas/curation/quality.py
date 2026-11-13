"""Wire models of ``GET /quality/readiness``: the dataset-level "ready to train" checklist."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ReadinessSeverity = Literal['ok', 'warn', 'block']
Verdict = Literal['ready', 'warn', 'blocked']


class ReadinessLink(BaseModel):
    """Where the check's work is: a project page and, for the Workspace, its address query."""

    section: str = Field(
        description='The project page: `workspace`, `export`, `quality/audit` or `classes`.'
    )
    query: str | None = Field(
        description='The search string of the page address (for the Workspace: `view`, `queue`, '
        'and the item filter keys it understands); null for none.'
    )


class ReadinessClassRow(BaseModel):
    """One class a check names: how many of what, and for the audit its precision."""

    class_name: str
    count: int = Field(description='What the check counts for this class.')
    precision: float | None = Field(
        default=None, description='Audited precision of the detector for this class, if audited.'
    )
    ci_low: float | None = Field(default=None, description='Wilson 95% interval, lower bound.')
    ci_high: float | None = Field(default=None, description='Wilson 95% interval, upper bound.')


class ReadinessCheck(BaseModel):
    id: str
    label: str = Field(description='The served name of the check.')
    severity: ReadinessSeverity
    message: str
    count: int | None = Field(description='The number the message is about; null when none.')
    link: ReadinessLink | None
    classes: list[ReadinessClassRow] = Field(
        description='The classes the message names (worst first); empty when none.'
    )


class ClassCoverage(BaseModel):
    """Items of one active class by how far a person has checked them."""

    class_name: str
    validated: int = Field(description='Checked by a person or an import.')
    machine_only: int = Field(description='A machine set the class (VLM, classifier), unchecked.')
    detector_only: int = Field(description='Only the detector named the class, unchecked.')
    test: int = Field(description='In the frozen test holdout.')


class DatasetReadiness(BaseModel):
    verdict: Verdict
    checks: list[ReadinessCheck]
    coverage: list[ClassCoverage]
    computed_at: str = Field(description='When the checks were computed (UTC, ISO 8601).')


__all__ = [
    'ClassCoverage',
    'DatasetReadiness',
    'ReadinessCheck',
    'ReadinessClassRow',
    'ReadinessLink',
    'ReadinessSeverity',
    'Verdict',
]
