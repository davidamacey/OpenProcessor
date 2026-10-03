"""Wire models of the ingest-policy routes (the policy shapes themselves live in
:mod:`src.services.curation.ingest_policy`)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.services.curation.ingest_policy import IngestPolicy, IngestPolicyBody


class IngestPolicyUpdate(IngestPolicyBody):
    """``PUT`` body: the policy plus the revision the caller read."""

    expected_revision: int = Field(ge=0)


class IngestPolicyPutResponse(IngestPolicy):
    # Class names the policy mentions that match no detector label or registry
    # class. Accepted (a model switch or a later class can make them valid).
    unknown_names: list[str] = Field(default_factory=list)


class PolicyPreviewClass(BaseModel):
    name: str = Field(description='The class NAME as stored / as the detector labels it.')
    would_embed: int
    would_not_embed: int


class IngestPolicyPreview(BaseModel):
    """What a candidate policy would embed, over the items already stored."""

    model_config = ConfigDict(extra='forbid')

    total_items: int
    scanned: int
    # True when the scan stopped at its cap; the counts then describe the scanned items.
    truncated: bool
    would_embed: int
    would_not_embed: int
    embedded_because_labeled: int = Field(
        description=(
            'Of would_embed, the items that embed only because a human or validated label '
            'always embeds; a fresh ingest of the same images has none of these.'
        )
    )
    estimated_vector_mb: float
    by_class: list[PolicyPreviewClass]
