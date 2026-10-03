"""The request and preview types of a combine-projects job (projects plan
section 6). ``extra='forbid'`` everywhere: a stale or misspelled key, or a
mapping row that is not a W10 ``ClassMappingEntry`` (``{to}`` / ``{drop}``),
is a 422."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.services.curation.dataset_import.mapping import (
    ClassMappingEntry,  # noqa: TC001 - pydantic field type, resolved at runtime
)
from src.services.issue_ids import stamp_issue_ids


MAX_SOURCES = 8

LabelStates = Literal['all', 'validated_only']
Dedup = Literal['content_hash', 'none']
HoldoutMode = Literal['preserve_union', 'recompute', 'none']


class CombineTarget(BaseModel):
    model_config = ConfigDict(extra='forbid')

    slug: str
    display_name: str = Field(min_length=1, max_length=200)
    description: str = ''


class CombineInclude(BaseModel):
    model_config = ConfigDict(extra='forbid')

    label_states: LabelStates = 'all'


class CombineSource(BaseModel):
    model_config = ConfigDict(extra='forbid')

    project: str
    include: CombineInclude = Field(default_factory=CombineInclude)


class CombineRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    target: CombineTarget
    sources: list[CombineSource] = Field(min_length=1)
    """List order is priority: the first source wins a label conflict and
    donates the image of a duplicate."""
    class_mapping: dict[str, list[ClassMappingEntry]] = Field(default_factory=dict)
    target_classes: list[str] | None = None
    dedup: Dedup = 'content_hash'
    dedup_iou: float = Field(default=0.9, gt=0.0, le=1.0)
    holdout: HoldoutMode = 'preserve_union'
    settings_from: str | None = None


class CombineStartRequest(CombineRequest):
    expected_preview_sha: str


class CombineIssue(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str = Field(
        default='',
        description='Unique within the response (`code[:project[:class]]`, `#2` on a repeat).',
    )
    code: str
    severity: Literal['error', 'warning'] = 'error'
    project: str | None = None
    message: str = ''
    detail: dict[str, Any] = Field(default_factory=dict)


def _issue_subject(issue: CombineIssue) -> str | None:
    parts = [issue.project, issue.detail.get('class')]
    return ':'.join(str(p) for p in parts if p) or None


class CombinePreview(BaseModel):
    ok: bool
    errors: list[CombineIssue]
    warnings: list[CombineIssue]
    preview_sha: str
    suggested_mapping: dict[str, list[ClassMappingEntry]]
    sources: list[dict[str, Any]]
    target: dict[str, Any]
    dedup: dict[str, Any]
    bytes: dict[str, int]

    @model_validator(mode='after')
    def _stamp_ids(self) -> Self:
        stamp_issue_ids([*self.errors, *self.warnings], _issue_subject)
        return self


__all__ = [
    'MAX_SOURCES',
    'CombineInclude',
    'CombineIssue',
    'CombinePreview',
    'CombineRequest',
    'CombineSource',
    'CombineStartRequest',
    'CombineTarget',
    'Dedup',
    'HoldoutMode',
    'LabelStates',
]
