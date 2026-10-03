"""Wire models for ``/open_vocab*``: the open-vocabulary prompt sets.

The body models carry the same defaults as the decoder's dataclasses
(:mod:`src.services.detection.open_vocab_set`), by importing its constants,
so a partial draft validates with informative issues rather than a pydantic
422. Ranges are checked by the validator, not by pydantic, for the same
reason.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.routers.curation._config_common_models import ActivateResponse, ActiveRef, ValidationReport
from src.services.detection.open_vocab_set import (
    DEFAULT_DEDUP_IOU,
    DEFAULT_IMAGE_MAX_SIDE,
    DEFAULT_MAX_AREA_FRAC,
    DEFAULT_MAX_ENABLED_TARGETS,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_MIN_AREA_FRAC,
    DEFAULT_MIN_SCORE,
    HitRateGate,
)


SetSourceWire = Literal['stored', 'template']
FieldType = Literal['string', 'int', 'float', 'bool', 'string_list']
FieldScope = Literal['set', 'target', 'gating', 'tier3_hit_rate']


class OpenVocabTargetBody(BaseModel):
    model_config = {'extra': 'allow'}

    prompt: str = ''
    class_name: str = ''
    min_score: float = DEFAULT_MIN_SCORE
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC
    max_area_frac: float = DEFAULT_MAX_AREA_FRAC
    max_instances: int = DEFAULT_MAX_INSTANCES
    parent_classes: list[str] = Field(default_factory=list)
    enabled: bool = True
    mask: bool = True


class OpenVocabHitRateBody(BaseModel):
    model_config = {'extra': 'allow'}

    enabled: bool = HitRateGate.enabled
    window: int = HitRateGate.window
    miss_threshold: int = HitRateGate.miss_threshold
    sample_floor: float = HitRateGate.sample_floor


class OpenVocabGatingBody(BaseModel):
    model_config = {'extra': 'allow'}

    tier2_vlm_precheck: bool = False
    tier3_hit_rate: OpenVocabHitRateBody = Field(default_factory=OpenVocabHitRateBody)


class OpenVocabBody(BaseModel):
    model_config = {'extra': 'allow'}

    display_name: str = ''
    targets: list[OpenVocabTargetBody] = Field(default_factory=list)
    image_max_side: int = DEFAULT_IMAGE_MAX_SIDE
    dedup_iou: float = DEFAULT_DEDUP_IOU
    run_on_ingest: bool = False
    max_enabled_targets: int = DEFAULT_MAX_ENABLED_TARGETS
    gating: OpenVocabGatingBody = Field(default_factory=OpenVocabGatingBody)


class OpenVocabSummary(BaseModel):
    name: str
    source: SetSourceWire
    read_only: bool
    revision: int | None
    etag: str
    display_name: str
    n_targets: int
    n_enabled_targets: int
    run_on_ingest: bool
    active: bool
    active_revision: int | None = None
    updated_at: str | None = None


class OpenVocabTemplateSummary(BaseModel):
    name: str
    source: Literal['template'] = 'template'
    read_only: Literal[True] = True
    path: str
    display_name: str | None = None
    n_targets: int


class OpenVocabList(BaseModel):
    sets: list[OpenVocabSummary]
    templates: list[OpenVocabTemplateSummary]
    active: ActiveRef
    config_revision: int
    stale: bool = False


class OpenVocabDoc(BaseModel):
    name: str
    source: SetSourceWire
    read_only: bool
    revision: int | None
    etag: str
    description: str
    body: OpenVocabBody
    created_at: str | None = None
    updated_at: str | None = None
    cloned_from: str | None = None
    active: bool = False
    active_revision: int | None = None
    validation: ValidationReport | None = None


class OpenVocabCreateRequest(BaseModel):
    name: str
    description: str = ''
    body: OpenVocabBody


class OpenVocabSaveRequest(BaseModel):
    expected_revision: int
    description: str | None = None
    body: OpenVocabBody


class OpenVocabCloneRequest(BaseModel):
    new_name: str
    revision: int | None = None
    source: SetSourceWire | None = None
    description: str | None = None
    from_project: str | None = None


class OpenVocabValidateRequest(BaseModel):
    name: str | None = None
    body: OpenVocabBody


class OpenVocabRevisionSummary(BaseModel):
    revision: int
    saved_at: str | None
    cloned_from: str | None
    description: str


class OpenVocabRevisionsResponse(BaseModel):
    name: str
    revisions: list[OpenVocabRevisionSummary]


class OpenVocabActivateRequest(BaseModel):
    revision: int | None = None
    expected_active: ActiveRef | None = None
    force: bool = False


class OpenVocabActivateResponse(ActivateResponse):
    pass


class OpenVocabRollbackRequest(BaseModel):
    expected_active: ActiveRef | None = None


class OpenVocabDeactivateRequest(BaseModel):
    expected_active: ActiveRef | None = None


class OpenVocabFieldSchema(BaseModel):
    scope: FieldScope
    field: str
    label: str
    type: FieldType
    default: Any
    min: float | None = None
    max: float | None = None
    advanced: bool = False
    help: str = ''


class OpenVocabSchema(BaseModel):
    fields: list[OpenVocabFieldSchema]
    max_enabled_targets_ceiling: int
