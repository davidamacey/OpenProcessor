"""Wire models for ``POST /ingest/bulk/preview`` and the preview block of the bulk config."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.routers.curation._ingest_sources_models import S3Location


class BulkPreviewRequest(BaseModel):
    """Exactly one of ``folder``, ``manifest``, ``s3`` and ``source_id``, as for a start.
    Nothing is ingested and nothing is written."""

    model_config = ConfigDict(extra='forbid')

    folder: str | None = Field(default=None, min_length=1)
    manifest: str | None = Field(default=None, min_length=1)
    s3: S3Location | None = None
    source_id: str | None = Field(default=None, min_length=1, max_length=64)
    extensions: list[str] | None = Field(default=None, max_length=32)

    @model_validator(mode='after')
    def _exactly_one_source(self) -> Self:
        chosen = [self.folder, self.manifest, self.s3, self.source_id]
        if sum(x is not None for x in chosen) != 1:
            raise ValueError('send exactly one of folder, manifest, s3 and source_id')
        return self


class PreviewExtensionWire(BaseModel):
    files: int
    bytes: int


class PreviewMegapixelsWire(BaseModel):
    # Headers read (no pixel decoded) from a bounded uniform sample of the files; 0 for S3.
    sampled: int
    unreadable: int
    # Bucket label ("<1", "1-3", "3-8", "8-16", "16-24", "24+" megapixels) to sampled count.
    histogram: dict[str, int]
    # The GPU-decode gate (OP_INGEST_GPU_MIN_MEGAPIXELS) the share is measured against.
    gate_megapixels: float
    share_at_or_above_gate: float | None = None


class PreviewEstimateWire(BaseModel):
    # megapixel_sample: one rate per image from the sampled sizes; rate_range: sizes
    # unknown, so the span from the slowest to the fastest measured route.
    basis: Literal['megapixel_sample', 'rate_range']
    duration_s_min: float
    duration_s_max: float
    vectors: float
    opensearch_bytes: int


class BulkPreview(BaseModel):
    project: str
    kind: Literal['folder', 'manifest', 's3']
    # What was previewed: the folder, manifest path or s3://bucket/prefix.
    target: str
    extensions: list[str]
    files: int
    total_bytes: int
    by_extension: dict[str, PreviewExtensionWire]
    megapixels: PreviewMegapixelsWire
    estimate: PreviewEstimateWire
    # True when a cap or a cancel ended the walk: every figure is then a lower bound.
    partial: bool
    stop_reason: Literal['file_cap', 'time_cap', 'cancelled'] | None = None
    unreadable_dirs: int
    unreadable_samples: list[str]
    symlinks_skipped: int
    # Manifest entries that do not exist / lie outside the source roots; files not statable.
    missing: int
    outside_roots: int
    unreadable_files: int
    gpu_decode: bool


class PreviewRateWire(BaseModel):
    route: Literal['cpu_small', 'cpu_large', 'gpu_large']
    images_per_s: float
    source_section: str
    measured_on: str


class PreviewMeasuredWire(BaseModel):
    value: float
    unit: str
    source_section: str
    measured_on: str


class BulkPreviewConfig(BaseModel):
    """The bounds of a preview and the measured figures its estimate uses, all from
    ``docs/PERFORMANCE.md`` (the section is named per figure)."""

    max_files: int
    max_seconds: float
    header_sample: int
    gpu_decode: bool
    gpu_min_megapixels: float
    rates: list[PreviewRateWire]
    embeddings_per_image: PreviewMeasuredWire
    opensearch_bytes_per_image: PreviewMeasuredWire
    source_document: str
