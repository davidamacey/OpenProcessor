"""Wire models for ``/ingest_profiles*``: named, revisioned ingest detector profiles."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from src.schemas.curation.config_common import ActivateResponse, ActiveRef, ValidationReport
from src.schemas.curation.ingest_profile import IngestProfileBody


ProfileSourceWire = Literal['stored', 'template']
FieldScope = Literal['primary', 'secondary']
FieldType = Literal['string', 'int', 'float']


class IngestProfileSummary(BaseModel):
    name: str
    source: ProfileSourceWire
    read_only: bool
    revision: int | None
    etag: str
    primary_model: str
    secondary_model: str | None = None
    active: bool
    active_revision: int | None = None
    updated_at: str | None = None


class IngestProfileTemplateSummary(BaseModel):
    name: str
    source: Literal['template'] = 'template'
    read_only: Literal[True] = True
    path: str
    primary_model: str


class IngestProfileList(BaseModel):
    profiles: list[IngestProfileSummary]
    templates: list[IngestProfileTemplateSummary]
    active: ActiveRef
    config_revision: int
    stale: bool = False


class IngestProfileDoc(BaseModel):
    name: str
    source: ProfileSourceWire
    read_only: bool
    revision: int | None
    etag: str
    description: str
    body: IngestProfileBody
    created_at: str | None = None
    updated_at: str | None = None
    cloned_from: str | None = None
    active: bool = False
    active_revision: int | None = None
    validation: ValidationReport | None = None


class IngestProfileCreateRequest(BaseModel):
    name: str
    description: str = ''
    body: IngestProfileBody


class IngestProfileSaveRequest(BaseModel):
    expected_revision: int
    description: str | None = None
    body: IngestProfileBody


class IngestProfileCloneRequest(BaseModel):
    new_name: str
    revision: int | None = None
    source: ProfileSourceWire | None = None
    description: str | None = None
    from_project: str | None = None


class IngestProfileValidateRequest(BaseModel):
    name: str | None = None
    body: IngestProfileBody


class IngestProfileRevisionSummary(BaseModel):
    revision: int
    saved_at: str | None
    cloned_from: str | None
    description: str


class IngestProfileRevisionsResponse(BaseModel):
    name: str
    revisions: list[IngestProfileRevisionSummary]


class IngestProfileActivateRequest(BaseModel):
    revision: int | None = None
    expected_active: ActiveRef | None = None
    force: bool = False


class IngestProfileActivateResponse(ActivateResponse):
    pass


class IngestProfileRollbackRequest(BaseModel):
    expected_active: ActiveRef | None = None


class IngestProfileDeactivateRequest(BaseModel):
    expected_active: ActiveRef | None = None


class IngestProfileFieldSchema(BaseModel):
    scope: FieldScope
    field: str
    label: str
    type: FieldType
    default: Any = None
    min: float | None = None
    max: float | None = None
    help: str = ''


class IngestProfileSchema(BaseModel):
    """One row per detector field, for both the ``primary`` and ``secondary``
    detector, so a form renders each control with its range. A field left null
    inherits the deployment's ``OP_INGEST_*`` value."""

    fields: list[IngestProfileFieldSchema]
