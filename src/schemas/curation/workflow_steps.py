"""Wire models of the curation loop: the steps, which are done, and what to do next."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


StepAction = Literal['adopt_detector_classes']

StepId = Literal[
    'ingest',
    'classes',
    'cluster',
    'activate',
    'label',
    'hotkeys',
    'near_dups',
    'export',
    'train',
    'review_disagreements',
    'audit',
]


class WorkflowStep(BaseModel):
    id: StepId
    label: str = Field(description='What to do, for example "Create classes from the detector".')
    detail: str = Field(description='A served sentence with the counts behind `done`.')
    done: bool
    section: str = Field(
        description='The project page that does it: `ingest`, `classes`, `workspace`, '
        '`quality/near-duplicates`, `export`, `train` or `audit`.'
    )
    query: str | None = Field(default=None, description='Optional search string for the link.')
    action: StepAction | None = Field(
        default=None,
        description='A write the step offers instead of a page: `adopt_detector_classes` is '
        '`POST /label_ops/adopt_detector`. Null when the step is done or has none.',
    )


class WorkflowSteps(BaseModel):
    steps: list[WorkflowStep] = Field(description='The loop, in order.')
    next: WorkflowStep | None = Field(description='The first step that is not done.')
    workspace_default_view: Literal['gallery', 'clusters'] = Field(
        description='`clusters` when an item partition serves, else `gallery`.'
    )
