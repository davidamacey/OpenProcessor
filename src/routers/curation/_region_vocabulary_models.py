"""Response model for ``GET {prefix}/regions/vocabulary``.

Typed so the OpenAPI contract (``contracts/openapi/curation.json``) carries
the shape a client renders from; the values come from
:func:`src.services.curation.region_vocabulary.region_vocabulary_catalog`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class VocabularyDetector(BaseModel):
    id: str
    label: str
    role: str
    filterable: bool = Field(
        description='True for values that can appear in stored region_detector.'
    )


class VocabularyEntry(BaseModel):
    id: str
    label: str
    role: str


class RejectionReasonEntry(BaseModel):
    """One pipeline-written ``region_rejection_reason`` value."""

    id: str = Field(description='The stored value (match=exact) or its prefix (match=prefix).')
    label: str
    kind: Literal['model_verdict', 'automatic', 'needs_human'] = Field(
        description=(
            'model_verdict: the verifier judged the box wrong; automatic: a '
            'geometry check rejected it; needs_human: no verdict was given.'
        )
    )
    match: Literal['exact', 'prefix']
    label_template: str | None = Field(
        default=None,
        description='For match=prefix: label with {detail} = the rest of the stored value.',
    )


class RegionProfileSummary(BaseModel):
    """The active region profile's identity, served on ``GET /health`` and
    ``GET /regions/vocabulary`` -- THE signal a client keys on to decide
    whether region-scoped UI/routes are available at all."""

    name: str
    display_name: str
    display_name_singular: str
    region_class_name: str
    text_reader: str


class RegionVocabularyResponse(BaseModel):
    region_profile: RegionProfileSummary | None = Field(
        description='The active region profile, or null when none is configured.'
    )
    detectors: list[VocabularyDetector]
    region_sources: list[VocabularyEntry]
    chain_actors: list[VocabularyEntry]
    text_rules: dict[str, Any] | None = Field(
        description='Which readings count as region text; null without a region profile.'
    )
    text_choices: list[str]
    rejection_reasons: list[RejectionReasonEntry]


__all__ = [
    'RegionProfileSummary',
    'RegionVocabularyResponse',
    'RejectionReasonEntry',
    'VocabularyDetector',
    'VocabularyEntry',
]
