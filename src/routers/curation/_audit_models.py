"""Wire models of the accuracy-audit routes (the arithmetic lives in
:mod:`src.services.curation.audit_math`)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.routers.curation._item_models import ItemDoc  # noqa: TC001 - pydantic
from src.services.curation.audit import DEFAULT_MIN_PER_CLASS, DEFAULT_SAMPLE_SIZE


class AuditStartRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    min_per_class: int = Field(
        default=DEFAULT_MIN_PER_CLASS,
        ge=1,
        le=1000,
        description='Crops to draw per detector class, and the audited count a class needs '
        'for the report to call its precision trustworthy.',
    )
    sample_size: int = Field(
        default=DEFAULT_SAMPLE_SIZE,
        ge=1,
        le=50_000,
        description='Total crops to draw (the human-time budget). Strata share it evenly '
        'toward min_per_class first.',
    )


class AuditStratum(BaseModel):
    detector_class: str
    available: int = Field(description='Eligible crops the detector gave this class.')
    sampled: int
    short_of_floor: bool = Field(
        description='True when the budget left this class fewer than min_per_class crops '
        'although it has them.'
    )


class AuditStartResponse(BaseModel):
    batch_id: str
    sampled: int
    requested: int
    min_per_class: int
    strata: list[AuditStratum]


class AuditQueueResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ItemDoc]


class AuditClassStat(BaseModel):
    name: str = Field(description='The class NAME.')
    n: int = Field(description='Audited crops.')
    correct: int
    precision: float | None
    ci_low: float = Field(description='Wilson 95% interval, lower bound.')
    ci_high: float
    insufficient_sample: bool = Field(description='n is below min_per_class.')


class AuditReport(BaseModel):
    audited: int = Field(description='Crops a human has labelled.')
    pending: int = Field(description='Drawn crops still waiting for a human.')
    min_per_class: int
    detector: list[AuditClassStat] = Field(
        description='Detector precision per detector class: the human agreed with its class.'
    )
    vlm: list[AuditClassStat] = Field(
        description='VLM precision per VLM class, over the crops whose label came from the VLM.'
    )
    confusion: dict[str, dict[str, int]] = Field(
        description='{detector class: {human class: crops}}.'
    )
    outcomes: dict[str, int] = Field(
        description='agree / detector_wrong / vlm_wrong / both_wrong counts.'
    )
