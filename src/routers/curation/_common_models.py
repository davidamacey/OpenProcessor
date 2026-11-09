"""Wire models for the image ingest routes, shared by ingest, ingest_upload and
ingest_objects. The other route groups keep theirs in sibling _*_models modules."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IngestImageRequest(BaseModel):
    # extra='forbid' so a wrong-key body (e.g. {'paths': [...]} instead
    # of the real field) 422s instead of silently validating to defaults and
    # ingesting nothing. Subclasses (IngestBatchItem) inherit this.
    model_config = ConfigDict(extra='forbid')

    path: str = Field(..., description='Absolute path to a JPEG on a mounted volume')
    source: str = Field(default='unknown', description='Source tag (e.g. hdd01, dataset_a)')


class IngestImageResponse(BaseModel):
    status: Literal['success', 'duplicate', 'failed']
    image_id: str = ''
    image_path: str
    imohash: str = ''
    n_crops: int = 0
    n_regions: int = 0
    # Of n_crops: items stored with / without a vector. n_not_embedded mixes policy
    # skips with encoder failures; n_embed_failed (embedding_state 'failed') counts
    # only the failures, which are a warning, not a success.
    n_embedded: int = 0
    n_not_embedded: int = 0
    n_embed_failed: int = 0
    # Detections the project's detect filter dropped (never stored; not in n_crops).
    n_filtered: int = 0
    error: str | None = None
    # A stable machine code alongside the message, e.g.
    # 'unservable_path', 'unsupported_type', 'decode_failed', 'too_large',
    # 'detector_infer', 'bulk_index' -- always present when status=='failed',
    # null otherwise. Lets a client group/label failures without parsing
    # `error`'s prose.
    error_kind: str | None = None
    # The client-supplied identifier for a byte-upload ingest, where
    # image_path is now the server-persisted path. Null for a server-path
    # ingest (image_path already IS the client-meaningful identifier).
    source_identifier: str | None = None
    # set when a configured secondary detector call failed for this
    # image (e.g. a Triton DEADLINE_EXCEEDED) -- the image still ingests
    # successfully on the primary detector's output alone, but a caller
    # asking for a secondary classifier needs to know it never ran.
    secondary_detector_error: str | None = None


class BatchIngestSummaryResponse(BaseModel):
    successful: int = 0
    duplicates: int = 0
    failed: int = 0
    crops_indexed: int = 0
    # count of images (among 'successful') where the configured
    # secondary detector call failed and was silently skipped before this
    # fix. Previously only a per-image 'warning' log line, invisible here.
    secondary_detector_failures: int = 0
    n_embedded: int = 0
    n_not_embedded: int = 0
    n_embed_failed: int = 0
    n_filtered: int = 0


class BatchIngestResponse(BaseModel):
    status: Literal['success', 'partial', 'error']
    summary: BatchIngestSummaryResponse
    results: list[IngestImageResponse] = Field(default_factory=list)
