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
from src.services.detection.open_vocab_select import DropReason  # noqa: TC001 - pydantic field type
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
from src.services.detection.segmenter_gate import GateReason  # noqa: TC001 - pydantic field type


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


class SegmenterAvailability(BaseModel):
    """The segmenter every open-vocabulary run needs. ``configured``: a
    segmenter URL is set; ``reachable``: it answered its health probe and has
    its model loaded (never true when not configured). The same fact
    ``GET /models/status`` reports in its segmenter row."""

    configured: bool
    reachable: bool


class OpenVocabList(BaseModel):
    sets: list[OpenVocabSummary]
    templates: list[OpenVocabTemplateSummary]
    active: ActiveRef
    config_revision: int
    stale: bool = False
    segmenter: SegmenterAvailability


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


class VocabularyOption(BaseModel):
    value: str
    label: str


class OpenVocabVocabulary(BaseModel):
    """Served labels for the pass's closed value sets: every value of the wire
    enums ``OpenVocabStatus`` (an image's ``open_vocab_status``, the reprocess
    filter), ``DropReason`` (a test hit's ``drop_reason``) and ``GateReason``
    (a test gate skip's ``reason``)."""

    statuses: list[VocabularyOption]
    drop_reasons: list[VocabularyOption]
    gate_reasons: list[VocabularyOption]


class OpenVocabSchema(BaseModel):
    fields: list[OpenVocabFieldSchema]
    max_enabled_targets_ceiling: int
    vocabulary: OpenVocabVocabulary


class OpenVocabTestRequest(BaseModel):
    """Run one UNSAVED target on one image; exactly one of ``image_id`` (a
    stored image) and ``image_base64`` (an uploaded JPEG/PNG)."""

    image_id: str | None = Field(
        default=None,
        description=(
            "A stored image's id: the item wire's `image_id`, the same id "
            '`POST /images/{image_id}/reprocess` takes.'
        ),
    )
    image_base64: str | None = None
    target: OpenVocabTargetBody
    image_max_side: int = DEFAULT_IMAGE_MAX_SIDE
    dedup_iou: float = DEFAULT_DEDUP_IOU
    gating: OpenVocabGatingBody = Field(default_factory=OpenVocabGatingBody)
    """Which gate tiers to apply. Tier 3 (hit-rate history) is never applied
    to a test: it has no history for an unsaved target."""


class OpenVocabTestHit(BaseModel):
    """One raw segmenter candidate, with what selection did with it. Boxes and
    outlines are normalized to the image: the client draws them."""

    bbox_norm: list[float]
    score: float
    mask_polygon: list[list[float]] | None = None
    selected: bool
    drop_reason: DropReason | None = None


class OpenVocabTestImage(BaseModel):
    width: int
    height: int


class OpenVocabTestGate(BaseModel):
    """What the gate decided for the target: ``run`` true, or a skip with the
    ``tier`` (1 registry rules, 2 vision-model pre-check) and ``reason``."""

    run: bool
    tier: int | None = None
    reason: GateReason | None = None


class OpenVocabTestResponse(BaseModel):
    image: OpenVocabTestImage
    prompt: str
    class_name: str
    gate: OpenVocabTestGate
    hits: list[OpenVocabTestHit]
    elapsed_ms: float
    validation: ValidationReport
