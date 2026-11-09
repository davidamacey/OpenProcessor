"""Wire models for the ingest config, region-drain verdict, status and path-lookup routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Runtime import: pydantic resolves the Literal annotation from module globals.
from src.services.curation.ingest_policy import IngestPolicy


class _PathLookupRequest(BaseModel):
    """Bulk path-existence check input. Capped client-side to 10k paths."""

    image_paths: list[str] = Field(..., max_length=10_000)


class _PathLookupResponse(BaseModel):
    """Maps known image_path -> existing image_id. Missing paths absent."""

    known_paths: dict[str, str]


# =============================================================================
# Typed ingest config / drain-verdict / status models.
# =============================================================================


class IngestUploadConfig(BaseModel):
    enabled: bool = True
    max_images_per_request: int
    max_bytes_per_request: int
    accepted_extensions: list[str]
    # False only ever for a deployment that hasn't wired an upload
    # root at all (there isn't one today -- upload_root always has a
    # default) -- kept as an explicit field rather than assumed true so a
    # future "uploads disabled" deployment mode can serve false here
    # without a wire-shape change.
    persists_bytes: bool = True


class IngestBatchConfig(BaseModel):
    enabled: bool = True
    max_items: int
    source_roots: list[str]


class IngestRegionDrainConfig(BaseModel):
    poll_interval_s: float
    stable_polls: int


class IngestDetectorLabel(BaseModel):
    class_id: int
    # The detector's own label, as stored in an item's ``proposal_name``.
    name: str
    # The registry class name this label maps to (``traffic light`` ->
    # ``traffic_light``); empty when the label has no slug-safe form.
    slug: str


class IngestDetectorInfo(BaseModel):
    """The deployment's ingest detector, read-only."""

    model: str
    version: str
    input_size: int
    assigns_class: bool
    # The confidence floor only applies when the detector assigns classes; a
    # proposer stores every detection the engine emits.
    confidence_floor_applies: bool
    n_labels: int
    labels: list[IngestDetectorLabel]


class IngestConfigResponse(BaseModel):
    upload: IngestUploadConfig
    batch: IngestBatchConfig
    region_drain: IngestRegionDrainConfig
    # Null when no ingest detector is configured (ingest answers 503).
    detector: IngestDetectorInfo | None = None
    # The project's detect filter and embedding policy (defaults when never set).
    policy: IngestPolicy = Field(default_factory=IngestPolicy)


class RegionDependencyStatusResponse(BaseModel):
    """One Triton model the active region profile depends on (V-1)."""

    role: str
    model: str
    ready: bool
    unavailable_since: str | None = None
    detail: str


class IngestRegionDrainResponse(BaseModel):
    pending_detection: int
    pending_verification: int
    total_unfinished: int
    # The stability verdict a walker used to have to invent a
    # window for client-side. drained=true only once total_unfinished
    # has read 0 for at least stable_polls consecutive polls (a single
    # zero reading right after a burst finishes could be a race, not a
    # drained queue).
    drained: bool
    stable_for_s: float
    observed_at: str
    # V-1: which of the active region profile's Triton models (detector /
    # segmenter) aren't READY right now, and since when. Empty when no
    # region profile is configured at all (the neutral/off case) -- not
    # populated as a false "stall".
    region_dependencies: list[RegionDependencyStatusResponse] = Field(default_factory=list)
    # A single human-readable line for the dashboard, e.g. "3516 item(s)
    # awaiting region detection; segmenter (sam3) unavailable since
    # 2026-09-25T14:02:11+00:00". None when there's nothing pending, or
    # every dependency is ready (queue is just working through a backlog,
    # not stalled).
    stall_reason: str | None = None


class IngestStatusResponse(BaseModel):
    total: int
    by_source: list[dict[str, Any]] = Field(default_factory=list)
    by_day: list[dict[str, Any]] = Field(default_factory=list)
