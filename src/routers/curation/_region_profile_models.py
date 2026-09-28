"""Wire models for ``/region_profiles*`` (W4, any_domain_plan.md §7.3)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config.settings import TritonModelConfig
from src.routers.curation._config_common_models import ActiveRef, ValidationReport


ProfileSourceWire = Literal['env', 'registered', 'stored', 'template']

FieldType = Literal[
    'string', 'int', 'float', 'bool', 'enum', 'string_list', 'int_list', 'float_pair', 'rgb'
]
ChoicesFrom = Literal[
    'detectors',
    'segmenters',
    'ocr_pipeline_models',
    'ocr_det_models',
    'ocr_rec_models',
    'registry_classes',
    'text_reader_modes',
]
AppliesWhen = Literal['detector', 'segmenter', 'reads_text', 'text_hint']


class RegionProfileBody(BaseModel):
    """Every :class:`~src.config.DetectionProfile` field except ``name``,
    JSON-typed (tuples/frozensets round-trip as lists). Defaults mirror
    the dataclass so a partial draft validates with informative issues
    rather than a pydantic 422."""

    model_config = {'extra': 'allow'}

    detector_model: str = ''
    detector_version: str = '1'
    input_size: int = 640
    confidence_floor: float = 0.4
    batch_limit: int = 16
    region_nms_iou: float = 0.5
    max_regions_per_item: int = 1
    letterbox_fill: list[int] = Field(default_factory=lambda: [114, 114, 114])
    feature_output: str = 'sppf_feat'
    aspect_min: float = 1.2
    aspect_max: float = 8.0
    text_hint_aspect_min: float = 1.5
    text_hint_aspect_max: float = 7.0
    text_hint_rec_floor: float = 0.70
    text_hint_len_min: int = 4
    text_hint_len_max: int = 10
    text_hint_enabled: bool = True
    text_hint_require_letters_and_digits: bool = False
    auto_confirm_aspect: list[float] = Field(default_factory=lambda: [0.5, 7.0])
    auto_confirm_area_frac: list[float] = Field(default_factory=lambda: [0.001, 0.40])
    text_pattern: str = r'[A-Z0-9 -]{2,}'
    segmenter_name: str = 'sam3'
    segmenter_version: str = '1'
    human_detector_name: str = 'human'
    human_detector_version: str = '1'
    ocr_det_model: str = 'paddleocr_det_trt'
    ocr_det_version: str = '1'
    ocr_det_input_size: int = 640
    ocr_det_prob_floor: float = 0.30
    ocr_rec_model: str = 'paddleocr_rec_trt'
    ocr_rec_version: str = '1'
    ocr_pipeline_model: str = TritonModelConfig.OCR_PIPELINE_MODEL
    text_reader: str = 'vlm_then_ocr'
    text_crop_margin: float = 0.05
    text_crop_min_height: int = 56
    text_min_height_ratio: float = 0.6
    text_border_margin: float = 0.08
    text_uppercase: bool = False
    text_charset: str = ''
    text_join: str = ' '
    text_len_min: int = 1
    text_len_max: int = 0
    text_stopwords: list[str] = Field(default_factory=list)
    text_min_confidence: float = 0.0
    text_format: str = ''
    text_placeholders: list[str] = Field(default_factory=list)
    text_reject_sequences: bool = False
    segmenter_text_prompt: str = ''
    secondary_shape_groups: list[str] = Field(default_factory=list)
    class_ids: list[int] = Field(default_factory=list)
    parent_classes: list[str] = Field(default_factory=list)
    assigns_class: bool = False
    labels_path: str = ''
    region_class_name: str = ''
    display_name: str = ''
    display_name_singular: str = ''


class RegionProfileEffective(BaseModel):
    reads_text: bool
    text_hint_active: bool
    legs: list[str]
    segmenter_enabled: bool


class RegionProfileSummary(BaseModel):
    name: str
    source: ProfileSourceWire
    read_only: bool
    revision: int | None
    etag: str
    display_name: str
    display_name_singular: str
    region_class_name: str
    text_reader: str
    reads_text: bool
    detector_model: str
    segmenter_text_prompt: str
    parent_classes: list[str]
    max_regions_per_item: int
    active: bool
    active_revision: int | None = None
    updated_at: str | None = None


class RegionProfileTemplateSummary(BaseModel):
    name: str
    source: Literal['template'] = 'template'
    read_only: Literal[True] = True
    path: str
    display_name: str | None = None
    reads_text: bool | None = None


class RegionProfileList(BaseModel):
    profiles: list[RegionProfileSummary]
    templates: list[RegionProfileTemplateSummary]
    active: ActiveRef
    config_revision: int
    stale: bool = False


class RegionProfileDoc(BaseModel):
    name: str
    source: ProfileSourceWire
    read_only: bool
    revision: int | None
    etag: str
    description: str
    body: RegionProfileBody
    effective: RegionProfileEffective
    created_at: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None
    cloned_from: str | None = None
    active: bool = False
    active_revision: int | None = None
    validation: ValidationReport | None = None


class RegionProfileCreateRequest(BaseModel):
    name: str
    description: str = ''
    body: RegionProfileBody


class RegionProfileCloneRequest(BaseModel):
    new_name: str
    revision: int | None = None
    source: ProfileSourceWire | None = None
    description: str | None = None
    from_project: str | None = None


class RegionProfileSaveRequest(BaseModel):
    expected_revision: int
    description: str | None = None
    body: RegionProfileBody


class RegionProfileValidateRequest(BaseModel):
    name: str | None = None
    body: RegionProfileBody


class RegionProfileRevisionSummary(BaseModel):
    revision: int
    saved_at: str | None
    cloned_from: str | None
    description: str


class RegionProfileRevisionsResponse(BaseModel):
    name: str
    revisions: list[RegionProfileRevisionSummary]


class RegionProfileActivateRequest(BaseModel):
    revision: int | None = None
    expected_active: ActiveRef | None = None
    force: bool = False


class RegionProfileRollbackRequest(BaseModel):
    expected_active: ActiveRef | None = None


class RegionProfileDeactivateRequest(BaseModel):
    expected_active: ActiveRef | None = None


class SegmenterPromptValidateRequest(BaseModel):
    text_prompt: str
    sole_leg: bool = True


class RegionProfileChoice(BaseModel):
    id: str
    label: str


class RegionProfileFieldSchema(BaseModel):
    field: str
    label: str
    group: str
    type: FieldType
    default: Any
    min: float | None = None
    max: float | None = None
    enum: list[RegionProfileChoice] | None = None
    advanced: bool = False
    applies_when: AppliesWhen | None = None
    choices_from: ChoicesFrom | None = None
    empty_choice: RegionProfileChoice | None = None
    help: str = ''


class RegionProfileGroup(BaseModel):
    id: str
    label: str


class RegionProfileSchema(BaseModel):
    fields: list[RegionProfileFieldSchema]
    groups: list[RegionProfileGroup]


__all__ = [
    'ActiveRef',
    'AppliesWhen',
    'ChoicesFrom',
    'FieldType',
    'ProfileSourceWire',
    'RegionProfileActivateRequest',
    'RegionProfileBody',
    'RegionProfileChoice',
    'RegionProfileCloneRequest',
    'RegionProfileCreateRequest',
    'RegionProfileDeactivateRequest',
    'RegionProfileDoc',
    'RegionProfileEffective',
    'RegionProfileFieldSchema',
    'RegionProfileGroup',
    'RegionProfileList',
    'RegionProfileRevisionSummary',
    'RegionProfileRevisionsResponse',
    'RegionProfileRollbackRequest',
    'RegionProfileSaveRequest',
    'RegionProfileSchema',
    'RegionProfileSummary',
    'RegionProfileTemplateSummary',
    'RegionProfileValidateRequest',
    'SegmenterPromptValidateRequest',
    'ValidationReport',
]
