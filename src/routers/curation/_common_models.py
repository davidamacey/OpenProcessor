"""Request/response Pydantic models for the curation routers.

Split out of ``_common.py`` to stay under the 700-LOC ratchet. Sub-modules
import these from ``src.routers.curation._common`` (which re-exports every
name here) rather than from this module directly, so this split is an
implementation detail, not a new import surface.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config.region_state import HUMAN_WRITABLE_STATUSES
from src.routers.curation._region_vocabulary_models import RegionProfileSummary  # noqa: TC001

# Runtime import: pydantic resolves the Literal annotation from module globals.
from src.services.curation.class_sources import HumanLabelSource  # noqa: TC001
from src.services.curation.label_import import DEFAULT_LABEL_SOURCE as _DEFAULT_LABEL_SOURCE


# =============================================================================
# Pydantic models (all request/response models for the curation router live
# here so sub-modules can import them without forming cycles).
# =============================================================================


class IngestImageRequest(BaseModel):
    path: str = Field(..., description='Absolute path to a JPEG on a mounted volume')
    source: str = Field(default='unknown', description='Source tag (e.g. hdd01, dataset_a)')


class IngestImageResponse(BaseModel):
    status: Literal['success', 'duplicate', 'failed']
    image_id: str = ''
    image_path: str
    imohash: str = ''
    n_crops: int = 0
    n_regions: int = 0
    error: str | None = None
    # BA-7: a stable machine code alongside the message, e.g.
    # 'unservable_path', 'unsupported_type', 'decode_failed', 'too_large',
    # 'detector_infer', 'bulk_index' -- always present when status=='failed',
    # null otherwise. Lets a client group/label failures without parsing
    # `error`'s prose.
    error_kind: str | None = None
    # BA-1: the client-supplied identifier for a byte-upload ingest, where
    # image_path is now the server-persisted path. Null for a server-path
    # ingest (image_path already IS the client-meaningful identifier).
    source_identifier: str | None = None


class BatchIngestSummaryResponse(BaseModel):
    successful: int = 0
    duplicates: int = 0
    failed: int = 0
    mismatches: int = 0
    missed_labels: int = 0
    unmatched_detections: int = 0
    labels_imported: int = 0
    crops_indexed: int = 0


class BatchIngestResponse(BaseModel):
    status: Literal['success', 'partial', 'error']
    summary: BatchIngestSummaryResponse
    results: list[IngestImageResponse] = Field(default_factory=list)
    # Populated only when the request set ``detect_mismatches``: one record
    # per model-vs-label disagreement (``kind`` = class_mismatch |
    # missed_label | unmatched_detection).
    disagreements: list[dict[str, Any]] = Field(default_factory=list)


class ImportLabelsRequest(BaseModel):
    image_path: str
    label_txt_path: str
    # Defaulted from the label importer so the public API carries no
    # deployment-specific label-source vocabulary.
    label_source: str = _DEFAULT_LABEL_SOURCE
    detect_mismatches: bool = False


class ImportLabelsBatchRequest(BaseModel):
    items: list[ImportLabelsRequest]


class CropLabelRequest(BaseModel):
    class_id: int
    # A human source only; the server writes class_source='human' itself.
    label_source: HumanLabelSource = 'human'


class CropBatchLabelRequest(BaseModel):
    crop_ids: list[str] = Field(..., max_length=5000)
    class_id: int
    label_source: HumanLabelSource = 'human'


class CropMoveRequest(BaseModel):
    crop_ids: list[str] = Field(..., max_length=5000)
    cluster_id: int


class CropExcludeRequest(BaseModel):
    """Exclude crops from training + clustering (reversible).

    Blurry / unidentifiable / partial crops the human doesn't want in
    the training set. ``reason`` defaults to ``'ignore'``; the UI can
    pass a more specific tag (``'blurry'``, ``'unidentifiable'``,
    ``'not_a_vehicle'``, ``'partial_crop'``) when the operator wants to
    record why (e.g. a whole cluster of blurry items).
    """

    crop_ids: list[str] = Field(..., max_length=5000)
    reason: str = 'ignore'


class CropUnexcludeRequest(BaseModel):
    """Reverse an exclusion (the labeler's Undo path for Ignore)."""

    crop_ids: list[str] = Field(..., max_length=5000)


class CropUndoBatchRequest(BaseModel):
    """Undo the most recent human class write on each crop."""

    crop_ids: list[str] = Field(..., max_length=5000)


class CropDiscardRequest(BaseModel):
    """Discard an item: clear its class (it doesn't belong where it is),
    dismiss it from every review queue, or both. Recorded like a label
    write, so ``POST /crops/{id}/label/undo`` reverses it."""

    model_config = {'extra': 'forbid'}

    clear_class: bool = True
    dismiss_from_review: bool = False


class CropDiscardBatchRequest(CropDiscardRequest):
    crop_ids: list[str] = Field(..., max_length=5000)


class ItemRegionRequest(BaseModel):
    """Set or clear the region-of-interest sub-bbox on a single item.

    ``frame`` says which frame ``region_bbox_norm`` is in: ``'source'``
    (the source image, the stored frame) or ``'parent'`` (the item crop;
    the server projects it through the item's own ``bbox_norm``).
    ``None`` clears the box and marks the item
    ``region_status='no_region_visible'`` (a deliberate human decision,
    distinct from "not yet detected").
    """

    model_config = {'extra': 'forbid'}

    region_bbox_norm: tuple[float, float, float, float] | None
    region_label_source: str = 'human'
    frame: Literal['source', 'parent'] = 'source'


class ItemBatchRegionRequest(BaseModel):
    """Bulk variant of ItemRegionRequest (e.g. "mark these N items as no
    region present")."""

    model_config = {'extra': 'forbid'}

    crop_ids: list[str] = Field(..., max_length=5000)
    region_bbox_norm: tuple[float, float, float, float] | None
    region_label_source: str = 'human'
    # 'parent' boxes are projected through each item's own bbox_norm.
    frame: Literal['source', 'parent'] = 'source'


# Region status values an operator may write — the lifecycle entries marked
# human_writable in src/config/region_state.py (stdlib-only, so the TS
# contract codegen exports the same set without importing the app).
HUMAN_REGION_STATUS_VALUES = HUMAN_WRITABLE_STATUSES


class CropBatchStatusRequest(BaseModel):
    """Bulk-set ``region_status`` over many items (the cluster-view triage op).

    Lets an operator select an outlier sub-cluster and mark every region
    ``false_positive`` / ``no_region_visible`` in one call, or bulk-confirm
    good regions (``region_status='detected'``).
    ``region_status`` must be one of ``HUMAN_REGION_STATUS_VALUES``.
    ``region_verified`` is accepted but ignored: the server derives it
    from ``region_status``.
    """

    model_config = {'extra': 'forbid'}

    crop_ids: list[str] = Field(..., max_length=5000)
    region_status: str
    region_verified: bool | None = Field(
        default=None, deprecated=True, description='Ignored; derived from region_status.'
    )
    region_label_source: str = 'human'


class ItemRegionMetaRequest(BaseModel):
    """Patch region metadata without touching ``region_bbox_norm``.

    Use this for operator corrections like fixing a region's OCR text or
    changing the status to ``verify_rejected``. To set or clear the bbox
    itself, use ``PUT /crops/{crop_id}/region``.

    Every field is optional; only the provided ones are written. ``None``
    on ``region_text`` clears the text, on ``region_rejection_reason``
    clears the reason. ``region_status`` must be one of
    ``HUMAN_REGION_STATUS_VALUES`` when present.
    """

    # extra='forbid' so a stale key 422s instead of silently no-opping;
    # model_fields_set distinguishes ``region_text=None`` (clear) from
    # "not in payload".
    model_config = {'extra': 'forbid'}

    region_text: str | None = None
    region_status: str | None = None
    region_rejection_reason: str | None = None
    region_label_source: str = 'human'


class TestHoldoutFreezeRequest(BaseModel):
    percent: int = Field(default=10, ge=1, le=50)
    # No longer used: selection is deterministic (SHA1-of-crop_id per
    # class, see src/services/curation/test_holdout.py) — a seeded RNG
    # can't guarantee a min-5-per-class floor or reproduce without
    # recording the seed. Kept accepted-but-ignored for backward
    # compatibility with existing callers.
    seed: int = 42


class TestHoldoutFreezeResponse(BaseModel):
    n_frozen: int
    n_classes_covered: int
    test_holdout_sha: str
    per_class_counts: dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded', 'down']
    triton: dict[str, Any]
    opensearch: dict[str, Any]
    vlm: dict[str, Any]
    registry: dict[str, Any]
    region_profile: RegionProfileSummary | None = Field(
        description=(
            'The active region profile, or null when none is configured -- '
            'THE signal a client checks to decide whether region-scoped '
            'UI/routes are available.'
        )
    )
    mlflow_public_url: str | None = Field(
        default=None,
        description=(
            'T1: the browser-reachable MLflow base URL (CurationConfig.'
            'mlflow_public_url / OP_MLFLOW_PUBLIC_URL), null when unset. '
            'A served train run may carry its own mlflow_run_url that is '
            'correct host-side but unreachable from an operator browser -- '
            'a client should build the link from this base rather than '
            'guessing a port.'
        ),
    )


class ExportYoloRequest(BaseModel):
    export_dir: str | None = None
    # Free-form version tag (e.g. 'v7.0a'). Recorded in manifest.json only.
    version_tag: str = ''
    # RNG seed for the stratified split. Recording it in the manifest is what
    # makes an export re-derivable.
    seed: int = 42
    # Cap distinct source frames collected (representative sample for pipeline
    # tests). None = full export.
    max_images: int | None = None
    # Optional whole-frame near-dup cut (cosine on the images index's
    # secondary embedding). e.g. 0.98 collapses near-identical bursts to
    # one frame; None disables.
    dedup_threshold: float | None = None


class ExportSingleClassRequest(BaseModel):
    """Body for the narrowed single-class / class-subset dataset export.

    Positives come from either the items' own class-labeled boxes
    (``box_source='item'``) or their region-of-interest sub-annotation
    (``box_source='region'``). All ``detected`` positives and all human
    ``false_positive`` hard negatives are kept in full; ``empty_bg_ratio``
    adds a small sample of genuinely empty frames as a fraction of
    positives so the detector still sees some no-target images.
    """

    export_dir: str | None = None
    version_tag: str = ''
    # Registry class ids forming this export's vocabulary, IN ORDER — the
    # dense label id written into the .txt files is the index into this
    # list. Required for box_source='item'; an optional parent-class
    # filter for box_source='region' (empty = every item's region).
    class_ids: list[int] = Field(default_factory=list)
    # 'item' = each item's own bbox_norm; 'region' = its region-of-interest.
    box_source: Literal['item', 'region'] = 'item'
    # data.yaml class name for the single region class in region mode.
    region_class_name: str = 'region'
    # Directory name under the export root; also the manifest's default
    # dataset identity. Keeps this export's artifacts and its own
    # `current` symlink separate from the multi-class export root.
    profile_name: str = 'single_class'
    # RNG seed for the cap sample + the split. Recorded in the manifest,
    # which is what makes an export re-derivable.
    seed: int = 42
    skip_test_split: bool = False
    # Fraction of positives to add as target-free frames (0.1 == 1 per 10).
    empty_bg_ratio: float = 0.1
    # Sample at most this many positive frames; None == all.
    max_positive_images: int | None = None
    # Optional whole-frame near-dup cut (cosine on the images index's
    # secondary embedding). e.g. 0.98 collapses near-identical bursts to
    # one frame; None disables.
    dedup_threshold: float | None = None
    # 'whole_frame' (full source frame) or 'item_crop' (parent item crop
    # with the region re-projected) — for whole-image vs crop training
    # A/B. 'item_crop' requires box_source='region'.
    image_mode: Literal['whole_frame', 'item_crop'] = 'whole_frame'
    # Longest output side in px: 640 for rapid iteration, 1280 for the full run.
    img_max_side: int = 1280
    # False writes labels + artifacts only (fast dry run, no pixel copy).
    copy_images: bool = True


class StatusResponse(BaseModel):
    status: str
    detail: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class _PathLookupRequest(BaseModel):
    """Bulk path-existence check input. Capped client-side to 10k paths."""

    image_paths: list[str] = Field(..., max_length=10_000)


class _PathLookupResponse(BaseModel):
    """Maps known image_path -> existing image_id. Missing paths absent."""

    known_paths: dict[str, str]


# =============================================================================
# BA-2/BA-3/BA-6: typed ingest config / drain-verdict / status models.
# =============================================================================


class IngestUploadConfig(BaseModel):
    enabled: bool = True
    max_images_per_request: int
    max_bytes_per_request: int
    accepted_extensions: list[str]
    # BA-1: false only ever for a deployment that hasn't wired an upload
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


class IngestConfigResponse(BaseModel):
    upload: IngestUploadConfig
    batch: IngestBatchConfig
    region_drain: IngestRegionDrainConfig


class IngestRegionDrainResponse(BaseModel):
    pending_detection: int
    pending_verification: int
    total_unfinished: int
    # BA-3: the stability verdict a walker used to have to invent a
    # window for client-side. drained=true only once total_unfinished
    # has read 0 for at least stable_polls consecutive polls (a single
    # zero reading right after a burst finishes could be a race, not a
    # drained queue).
    drained: bool
    stable_for_s: float
    observed_at: str


class IngestStatusResponse(BaseModel):
    total: int
    by_source: list[dict[str, Any]] = Field(default_factory=list)
    by_day: list[dict[str, Any]] = Field(default_factory=list)


class CropFlagNewClassRequest(BaseModel):
    """Marks crops as needing a class that doesn't exist in the registry
    yet — for batch curator review (typically weekly)."""

    crop_ids: list[str] = Field(..., max_length=5000)
    note: str = ''


class CurationSettingsResponse(BaseModel):
    """``GET,PUT /curation/settings`` response envelope.

    ``defaults`` is deliberately ``dict[str, str]`` (an OPEN map keyed by
    axis id), not a fixed set of named fields (``cluster``/``sort``/etc.)
    -- a future axis must not require a wire-format change. Missing key =
    no shared override for that axis; the caller falls back to
    ``GET /methods``'s own hardcoded-default resolution (see
    ``src.services.curation.strategy_registry.resolve_effective_default``).
    """

    defaults: dict[str, str] = Field(default_factory=dict)
    updated_at: str | None = None
    updated_by: str | None = None


class CurationSettingsUpdateRequest(BaseModel):
    """``PUT /curation/settings`` body -- partial by design. Only the axes
    present here are validated + merged into the stored document; axes
    already set are left untouched (see
    ``src.clients.curation_opensearch.update_curation_settings``).

    A value of ``null`` for an axis clears that axis's shared override
    (falls back to that endpoint's own hardcoded default / each /review
    tab's own tuned sort) -- otherwise a pinned override was permanently
    unreachable once set, since every id must be currently-advertised and
    there was no way to express "go back to no override" (raised by the
    Cropwright settings-UI integration pass)."""

    defaults: dict[str, str | None] = Field(default_factory=dict)


class _PublishEvent(BaseModel):
    """Event-publish payload — used by out-of-process workers.

    ``extra='forbid'``: a publisher posting an unknown key (e.g. a storage
    field name instead of the wire ``region_status``) gets a 422 instead
    of an event silently stripped of its payload.
    """

    model_config = {'extra': 'forbid'}

    type: str
    crop_id: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    class_source: str | None = None
    region_status: str | None = None
    region_text: str | None = None
    image_path: str | None = None
    topic: str | None = None
    extra: dict[str, Any] | None = None
