"""Wire models of the cluster proposal, the scope check and the explicit apply."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClassVotes(BaseModel):
    class_name: str
    votes: int


class NewClassVotes(BaseModel):
    name: str
    votes: int


class AttributeValueCount(BaseModel):
    value: str
    count: int


class AttributeSummary(BaseModel):
    key: str
    values: list[AttributeValueCount]
    answered: int = Field(description='Answers that gave this attribute.')


class Proposal(BaseModel):
    """What the sampled answers say the scope is, never what it is. The label is
    proposed only when enough answers agree; ``apply`` is the one path that turns
    it into labels."""

    class_name: str | None = Field(
        description='The top class when the gate passes, else null (the distribution is still given).'
    )
    in_registry: bool = Field(description='Whether `class_name` is a registry class right now.')
    agreement: float = Field(description='Top votes over answered.')
    answered: int = Field(description='Sampled answers that named a class or proposed one.')
    min_agreement: float
    passes_gate: bool = Field(description='`answered >= 3` and `agreement >= min_agreement`.')
    distribution: list[ClassVotes]
    new_class_proposals: list[NewClassVotes]
    attribute_summary: list[AttributeSummary]


class ScopeNow(BaseModel):
    """The scope's members now against those the run resolved: a re-refine or a
    relabel changes them, and a proposal about other members is stale."""

    count: int
    overlap: int
    jaccard: float
    changed: bool


class AssistApplyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    class_name: str
    label_source: Literal['human'] = 'human'
    allow_changed_scope: bool = False
    dry_run: bool = True


class ApplyConflict(BaseModel):
    crop_id: str
    current_source: str | None = None


class AssistApplyResponse(BaseModel):
    dry_run: bool
    class_name: str
    selected: int = Field(description='Current members the class may be applied to.')
    skipped_locked: int = Field(
        description='Current members left alone: a human- or import-set class, or validated.'
    )
    updated: int = 0
    updated_ids: list[str] = Field(default_factory=list)
    conflicts: list[ApplyConflict] = Field(default_factory=list)
    scope_now: ScopeNow


__all__ = [
    'ApplyConflict',
    'AssistApplyRequest',
    'AssistApplyResponse',
    'AttributeSummary',
    'AttributeValueCount',
    'ClassVotes',
    'NewClassVotes',
    'Proposal',
    'ScopeNow',
]
