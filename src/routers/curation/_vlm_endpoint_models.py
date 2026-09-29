"""Wire models for ``/vlm/endpoints*`` and the local-model routes (W9,
any_domain_plan.md §7.8). Every request model is ``extra='forbid'``.

The stored body and the service-level probe record are defined in
``src/services/labeling/vlm_endpoint_body.py`` (pure pydantic, shared with
the worker); this module adds the typed wire shapes on top.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.routers.curation._config_common_models import (
    ActiveConfigResponse,
    ActiveRef,
    ValidationIssue,
    ValidationReport,
)
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody
from src.services.labeling.vlm_endpoints import (
    CATALOG_STATUS_LABELS,
    LOCALITY_LABELS,
    SOURCE_LABELS,
    STATUS_LABELS,
)


EndpointStatusWire = Literal['ready', 'unprobed', 'probe_failed', 'unreachable']
LocalityWire = Literal['compose', 'host', 'private', 'external', 'unknown']
SourceWire = Literal['env', 'stored']
ExternalPolicyWire = Literal['ack', 'deny']
CatalogStatusWire = Literal['tested', 'to_verify']


class Choice(BaseModel):
    id: str | None
    label: str


class VlmProbeResult(BaseModel):
    ok: bool
    probed_at: str
    latency_ms: float | None = None
    models_listed: list[str] = []
    model_listed: bool | None = None
    root: str | None = None
    max_model_len: int | None = None
    vision_ok: bool | None = None
    json_mode_supported: bool | None = None
    reasoning_channel: bool | None = None
    image_tokens: int | None = None
    max_images_ok: bool | None = None
    issues: list[ValidationIssue] = []


class VlmEndpointSummary(BaseModel):
    name: str
    source: SourceWire
    read_only: bool
    revision: int | None
    etag: str
    description: str
    base_url: str
    model: str
    catalog_id: str | None
    locality: LocalityWire | None
    sends_images_externally: bool
    warning: str | None
    api_key_ref: str | None
    api_key_present: bool
    status: EndpointStatusWire
    last_probe_at: str | None
    #: Slugs of the projects whose VLM is this endpoint (each project's
    #: activation is its own; the registry is deployment-wide).
    active_in: list[str] = []
    updated_at: str | None = None


class SecretRef(BaseModel):
    ref: str
    present: bool
    choice: Choice


class VlmEndpointLabels(BaseModel):
    status: dict[str, str]
    locality: dict[str, str]
    source: dict[str, str]


class VlmEndpointList(BaseModel):
    endpoints: list[VlmEndpointSummary]
    config_revision: int
    stale: bool = False
    external_policy: ExternalPolicyWire
    secret_refs: list[SecretRef]
    labels: VlmEndpointLabels


class VlmEndpointDoc(BaseModel):
    name: str
    source: SourceWire
    read_only: bool
    revision: int | None
    etag: str
    description: str
    body: VlmEndpointBody
    api_key_present: bool
    locality: LocalityWire | None
    sends_images_externally: bool
    warning: str | None
    last_probe: VlmProbeResult | None = None
    created_at: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None
    cloned_from: str | None = None
    active_in: list[str] = []
    validation: ValidationReport | None = None


class VlmEndpointCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    description: str = ''
    body: VlmEndpointBody


class VlmEndpointCloneRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    new_name: str
    revision: int | None = None
    description: str | None = None


class VlmEndpointSaveRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected_revision: int
    description: str | None = None
    body: VlmEndpointBody


class VlmValidateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str | None = None
    body: VlmEndpointBody


class VlmValidateResponse(BaseModel):
    validation: ValidationReport
    locality: LocalityWire | None
    sends_images_externally: bool
    probe: VlmProbeResult | None = None


class VlmRevisionSummary(BaseModel):
    revision: int
    saved_at: str | None
    cloned_from: str | None
    description: str


class VlmRevisionsResponse(BaseModel):
    name: str
    revisions: list[VlmRevisionSummary]


class VlmActivateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    revision: int | None = None
    expected_active: ActiveRef | None = None
    force: bool = False
    acknowledge_external: bool = False


class VlmRollbackRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected_active: ActiveRef | None = None


class VlmDeactivateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected_active: ActiveRef | None = None


class VlmActiveResponse(ActiveConfigResponse):
    validation: ValidationReport | None = None


# --- schema rows (the form) -----------------------------------------------------------

VlmFieldType = Literal['string', 'int', 'float', 'bool', 'enum']
VlmChoicesFrom = Literal['secret_refs', 'vlm_catalog']


class VlmEndpointFieldSchema(BaseModel):
    field: str
    label: str
    group: str
    type: VlmFieldType
    default: Any
    min: float | None = None
    max: float | None = None
    enum: list[Choice] | None = None
    advanced: bool = False
    choices_from: VlmChoicesFrom | None = None
    empty_choice: Choice | None = None
    help: str = ''


class VlmEndpointGroup(BaseModel):
    id: str
    label: str


class VlmEndpointSchema(BaseModel):
    fields: list[VlmEndpointFieldSchema]
    groups: list[VlmEndpointGroup]


# --- catalog and local model --------------------------------------------------------


class VlmCatalogEntry(BaseModel):
    id: str
    choice: Choice
    hf_repo: str
    family: str
    license: str
    license_url: str
    gated: bool
    params_b: float | None
    quantization: str | None
    context_max: int
    max_model_len: int
    max_images: int
    vram_gb: float
    disk_gb: float | None
    status: CatalogStatusWire
    rank: int
    multi_box_verified: bool | None
    text_reading_verified: bool | None
    fits: bool | None
    serving: bool
    desired: bool


class VlmLocalServed(BaseModel):
    model: str | None
    root: str | None
    catalog_id: str | None
    max_model_len: int | None


class VlmLocalDesired(BaseModel):
    catalog_id: str
    requested_at: str | None
    command: str


class VlmLocalStatus(BaseModel):
    configured: bool
    endpoint: str | None
    served: VlmLocalServed | None
    desired: VlmLocalDesired | None
    restart_required: bool
    poll_after_s: int | None
    gpu_total_gb: float | None
    can_restart_from_api: bool
    reason: str


class VlmCatalogLabels(BaseModel):
    status: dict[str, str]


class VlmCatalogResponse(BaseModel):
    entries: list[VlmCatalogEntry]
    local: VlmLocalStatus
    labels: VlmCatalogLabels


class VlmLocalSelectRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    catalog_id: str
    force: bool = False


__all__ = [
    'CATALOG_STATUS_LABELS',
    'LOCALITY_LABELS',
    'SOURCE_LABELS',
    'STATUS_LABELS',
    'ActiveRef',
    'Choice',
    'SecretRef',
    'ValidationIssue',
    'ValidationReport',
    'VlmActivateRequest',
    'VlmActiveResponse',
    'VlmCatalogEntry',
    'VlmCatalogLabels',
    'VlmCatalogResponse',
    'VlmDeactivateRequest',
    'VlmEndpointBody',
    'VlmEndpointCloneRequest',
    'VlmEndpointCreate',
    'VlmEndpointDoc',
    'VlmEndpointFieldSchema',
    'VlmEndpointGroup',
    'VlmEndpointLabels',
    'VlmEndpointList',
    'VlmEndpointSaveRequest',
    'VlmEndpointSchema',
    'VlmEndpointSummary',
    'VlmLocalDesired',
    'VlmLocalSelectRequest',
    'VlmLocalServed',
    'VlmLocalStatus',
    'VlmProbeResult',
    'VlmRevisionSummary',
    'VlmRevisionsResponse',
    'VlmRollbackRequest',
    'VlmValidateRequest',
    'VlmValidateResponse',
]
