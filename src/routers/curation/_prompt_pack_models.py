"""Wire models for ``/prompt_packs*`` (W3, any_domain_plan.md §7.2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.routers.curation._config_common_models import (
    ActiveConfigResponse,
    ActiveRef,
    ValidationReport,
)
from src.services.labeling.vlm_prompts import REPLY_KEY_CONTRACT


PackSourceWire = Literal['builtin', 'file', 'template', 'stored']


class PromptPackBody(BaseModel):
    """Every :class:`~src.services.labeling.vlm_prompts.PromptPack` field
    except ``name``. Defaults to ``''``/``{}`` so a partial draft can be
    posted to ``/validate`` -- ``pack_validation.validate_pack`` reports
    missing content as issues, not a pydantic 422."""

    class_system: str = ''
    class_user_template: str = ''
    open_class_system: str = ''
    open_class_user_template: str = ''
    combined_system: str = ''
    combined_user_template: str = ''
    combined_batch_system: str = ''
    combined_batch_rules: str = ''
    region_system: str = ''
    region_user: str = ''
    region_batch_system: str = ''
    region_batch_user: str = ''
    region_visible_system: str = ''
    region_visible_user: str = ''
    class_descriptions: dict[str, str] = Field(default_factory=dict)
    synonyms: dict[str, str] = Field(default_factory=dict)


class PromptPackSummary(BaseModel):
    name: str
    source: PackSourceWire
    read_only: bool
    revision: int | None
    etag: str
    description: str
    asks_region_text: bool
    active: bool
    active_revision: int | None = None
    updated_at: str | None = None


class PromptPackTemplateSummary(BaseModel):
    name: str
    source: Literal['template'] = 'template'
    read_only: Literal[True] = True
    path: str
    display_name: str | None = None


class PromptPackList(BaseModel):
    packs: list[PromptPackSummary]
    templates: list[PromptPackTemplateSummary]
    active: ActiveRef
    config_revision: int
    stale: bool = False


class PromptPackDoc(BaseModel):
    name: str
    source: PackSourceWire
    read_only: bool
    revision: int | None
    etag: str
    description: str
    body: PromptPackBody
    created_at: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None
    cloned_from: str | None = None
    active: bool = False
    active_revision: int | None = None
    validation: ValidationReport | None = None


class PromptPackCreateRequest(BaseModel):
    name: str
    description: str = ''
    body: PromptPackBody


class PromptPackCloneRequest(BaseModel):
    new_name: str
    revision: int | None = None
    source: PackSourceWire | None = None
    description: str | None = None
    # W3: the one legitimate cross-project read in this wave -- clone a
    # pack out of another project's config store. Read-only bind on
    # ``from_project``; the write always lands in the bound (target)
    # project.
    from_project: str | None = None


class PromptPackSaveRequest(BaseModel):
    expected_revision: int
    description: str | None = None
    body: PromptPackBody


class PromptPackValidateRequest(BaseModel):
    name: str | None = None
    body: PromptPackBody


class PromptPackRevisionSummary(BaseModel):
    revision: int
    saved_at: str | None
    cloned_from: str | None
    description: str


class PromptPackRevisionsResponse(BaseModel):
    name: str
    revisions: list[PromptPackRevisionSummary]


class PromptPackActivateRequest(BaseModel):
    revision: int | None = None
    expected_active: ActiveRef | None = None
    force: bool = False


class PromptPackRollbackRequest(BaseModel):
    expected_active: ActiveRef | None = None


class PromptPackFieldSchema(BaseModel):
    field: str
    label: str
    group: str
    kind: Literal['text', 'map']
    formatted: bool
    required_placeholders: list[str]
    allowed_placeholders: list[str]
    expected_reply_keys: list[str]
    optional_reply_keys: list[str]
    used_by: list[str]
    help: str


class PromptPackPlaceholderHelp(BaseModel):
    name: str
    meaning: str
    example: str


class PromptPackCallSchema(BaseModel):
    id: str
    label: str
    fields: list[str]
    testable: bool = True


class PromptPackSchema(BaseModel):
    fields: list[PromptPackFieldSchema]
    placeholders: list[PromptPackPlaceholderHelp]
    calls: list[PromptPackCallSchema]
    reply_key_contract: dict[str, dict[str, list[str]]] = Field(
        default_factory=lambda: {
            call_id: {'required': c['required'], 'optional': c['optional']}
            for call_id, c in REPLY_KEY_CONTRACT.items()
        }
    )


__all__ = [
    'ActiveConfigResponse',
    'PackSourceWire',
    'PromptPackActivateRequest',
    'PromptPackBody',
    'PromptPackCallSchema',
    'PromptPackCloneRequest',
    'PromptPackCreateRequest',
    'PromptPackDoc',
    'PromptPackFieldSchema',
    'PromptPackList',
    'PromptPackPlaceholderHelp',
    'PromptPackRevisionSummary',
    'PromptPackRevisionsResponse',
    'PromptPackRollbackRequest',
    'PromptPackSaveRequest',
    'PromptPackSchema',
    'PromptPackSummary',
    'PromptPackTemplateSummary',
    'PromptPackValidateRequest',
]
