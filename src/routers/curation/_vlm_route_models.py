"""Request/response models of the VLM label/verify routes
(:mod:`src.routers.curation.vlm`)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VlmLabelBatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    crop_ids: list[str] = Field(..., min_length=1, max_length=5000)


class VlmVerifyRegionsRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    crop_ids: list[str] = Field(..., min_length=1, max_length=5000)


class VlmVerifyRegionBatchItem(BaseModel):
    """One item in a batched region-verify request.

    Mirrors the single-crop ``/curation/vlm/verify_region`` shape. The
    caller is responsible for cropping the sub-region out of its source
    crop and base64-encoding the JPEG bytes — the API does not re-derive
    the region JPEG from OpenSearch on this path so the batch endpoint
    can serve callers (e.g. a detection worker, training scripts) that
    already hold the JPEG in memory.
    """

    crop_id: str
    region_image_b64: str = Field(
        ...,
        description='Base64-encoded JPEG of the sub-region crop (no data: prefix).',
    )
    candidate_text: str | None = Field(
        default=None,
        description='Optional caller-supplied candidate text (e.g. from a text-detection '
        'pre-pass) echoed back on the result.',
    )


class VlmVerifyRegionBatchRequest(BaseModel):
    """Request body for ``POST /curation/vlm/verify_region_batch``."""

    model_config = ConfigDict(extra='forbid')

    items: list[VlmVerifyRegionBatchItem] = Field(..., min_length=1)


class VlmVerifyRegionBatchResult(BaseModel):
    """One result in the verify_region_batch response.

    A ``crop_id`` from the request that got no usable VLM answer at all
    (whole-chunk upstream failure, empty/unparseable/misaligned reply,
    or an individual crop missing from an otherwise-aligned reply) is
    absent from ``results`` entirely -- never emitted with a
    synthesized ``is_region=False``. Callers must treat a missing
    crop_id as "retry later", the same contract
    ``/vlm/region_visible_batch`` uses for its map.
    """

    crop_id: str
    is_region: bool
    confidence: str
    reason: str = ''
    candidate_text: str | None = None


class VlmVerifyRegionBatchResponse(BaseModel):
    results: list[VlmVerifyRegionBatchResult]


class VlmRegionVisibleBatchItem(BaseModel):
    crop_id: str
    image_b64: str = Field(
        ...,
        description='Base64-encoded JPEG of the item crop (no data: prefix).',
    )


class VlmRegionVisibleBatchRequest(BaseModel):
    """Request body for ``POST /curation/vlm/region_visible_batch``."""

    model_config = ConfigDict(extra='forbid')

    items: list[VlmRegionVisibleBatchItem] = Field(..., min_length=1)


class VlmRegionVisibleBatchResponse(BaseModel):
    """``{crop_id: bool}`` mapping — True means a sub-region is visible."""

    visible: dict[str, bool]
