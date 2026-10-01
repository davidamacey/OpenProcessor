"""Wire models for ``/datasets/*`` (W10.14). Request models live with the
service (:mod:`src.services.curation.dataset_import.options`); every
request model is ``extra='forbid'``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.routers.curation._dataset_issue_models import (
    DatasetIssueWire,  # noqa: TC001 - pydantic field type, resolved at runtime
)
from src.services.curation.dataset_import.options import (  # noqa: TC001 - runtime for pydantic
    DatasetImportOptions,
)


ImportStatus = Literal[
    'queued',
    'running',
    'paused_backpressure',
    'completed',
    'completed_with_errors',
    'failed',
    'cancelled',
    'interrupted',
    'undoing',
    'undone',
]
MapKind = Literal['item', 'region', 'skip']
Action = Literal['map', 'create', 'skip', 'region']


class MappingSuggestionWire(BaseModel):
    action: Action
    class_id: int | None = None
    class_name: str | None = None
    match: str


class ResolvedMapTarget(BaseModel):
    dataset_class: str
    kind: MapKind
    class_id: int | None = None
    class_name: str | None = None
    created: bool = False


class IndexWouldHaveMapped(BaseModel):
    class_id: int
    class_name: str


class DatasetClassRow(BaseModel):
    dataset_class: str
    dataset_id: int | None = None
    boxes: int
    images: int
    source_class_id: int | None = None
    index_would_have_mapped_to: IndexWouldHaveMapped | None = None
    suggestion: MappingSuggestionWire
    resolved: ResolvedMapTarget | None = None
    merged_from: list[str] = Field(default_factory=list)


class DatasetSplitRow(BaseModel):
    split: str
    images: int
    labeled: int
    negatives: int
    unlabeled: int
    boxes: int


class DatasetTotals(BaseModel):
    images: int
    boxes: int
    images_already_indexed: int
    images_to_ingest: int


class DatasetRegionInfo(BaseModel):
    profile: dict[str, Any] | None = None
    region_class_name: str | None = None
    parent_classes: list[str] = Field(default_factory=list)
    parents_mode: Literal['labels', 'detect']
    standalone_boxes: int = 0


class DatasetEstimate(BaseModel):
    detector_images: int
    embeddings: int


class DatasetPreview(BaseModel):
    project: str
    format: str
    root: str
    source_sha: str
    import_key: str
    op_export: dict[str, Any] | None = None
    splits: list[DatasetSplitRow]
    totals: DatasetTotals
    classes: list[DatasetClassRow]
    region: DatasetRegionInfo | None = None
    issues: list[DatasetIssueWire]
    blocking: bool
    force_allowed: bool
    estimate: DatasetEstimate


class DatasetImportProgress(BaseModel):
    images_total: int = 0
    images_done: int = 0
    images_failed: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    images_per_s: float | None = None
    eta_s: float | None = None


class DisagreementsWire(BaseModel):
    counts: dict[str, int] = Field(default_factory=dict)
    samples: list[dict[str, Any]] = Field(default_factory=list)


class DatasetImportReportWire(BaseModel):
    images_created: int = 0
    images_reused: int = 0
    images_failed: int = 0
    images_skipped: int = 0
    items_created: int = 0
    items_updated: int = 0
    items_noop: int = 0
    labels_written: int = 0
    boxes_written: int = 0
    standalone_regions: int = 0
    negatives: int = 0
    unlabeled: int = 0
    parents_detected: int = 0
    proposals_created: int = 0
    proposals_merged: int = 0
    holdout_frozen: int = 0
    label_conflicts_locked: int = 0
    items_reconciled_removed: int = 0
    disagreements: DisagreementsWire = Field(default_factory=DisagreementsWire)


class NextStep(BaseModel):
    action: str
    method: str
    path: str
    reason: str


class DatasetImportSource(BaseModel):
    format: str | None = None
    root: str | None = None
    source_sha: str | None = None


class DatasetUndoReportWire(BaseModel):
    import_id: str
    dry_run: bool
    items_deleted: int = 0
    items_restored: int = 0
    items_reinstated: int = 0
    items_kept_human_edited: int = 0
    items_kept_shared: int = 0
    class_labels_removed: int = 0
    boxes_removed: int = 0
    boxes_kept_human_edited: int = 0
    proposals_deleted: int = 0
    holdout_flags_cleared: int = 0
    images_deleted: int = 0
    images_kept: int = 0
    classes_deprecated: list[str] = Field(default_factory=list)
    samples: dict[str, list[Any]] = Field(default_factory=dict)


class DatasetImportJob(BaseModel):
    project: str
    import_id: str
    import_key: str
    name: str
    status: ImportStatus
    reused: bool = False
    progress: DatasetImportProgress
    waiting_for: str | None = None
    report: DatasetImportReportWire
    mapping: list[ResolvedMapTarget] = Field(default_factory=list)
    options: DatasetImportOptions
    source: DatasetImportSource
    issues_summary: list[dict[str, Any]] = Field(default_factory=list)
    undo: DatasetUndoReportWire | None = None
    next_steps: list[NextStep] = Field(default_factory=list)
    error: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    poll_after_s: int | None = None
    labels: dict[str, dict[str, str]] = Field(default_factory=dict)


class DatasetImportList(BaseModel):
    items: list[DatasetImportJob]
    total: int
    page: int
    page_size: int


class DatasetIssuePage(BaseModel):
    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int


class DatasetImportEntry(BaseModel):
    source_stem: str
    rel_path: str
    image_id: str | None = None
    image_path: str | None = None
    image_created: bool | None = None
    split: str | None = None
    label_state: str | None = None
    status: str
    error_kind: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    boxes: list[dict[str, Any]] = Field(default_factory=list)


class DatasetImportEntryPage(BaseModel):
    items: list[DatasetImportEntry]
    total: int
    page: int
    page_size: int


class DatasetUndoRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    dry_run: bool = True
    remove_images: bool = True
    deprecate_created_classes: bool = True


class DatasetUploadResponse(BaseModel):
    upload_id: str
    dataset_path: str
    bytes: int
    files: int


class DatasetFormatInfo(BaseModel):
    format: str
    label: str


class DatasetIssueCatalogEntry(BaseModel):
    code: str
    severity: str
    blocking: bool
    bypassable: bool
    label: str


class LabeledChoice(BaseModel):
    value: str
    label: str
    description: str = ''


class DatasetUploadLimits(BaseModel):
    max_bytes: int
    max_files: int
    ttl_hours: int
    preview_max_files: int


class DatasetFormatsResponse(BaseModel):
    formats: list[DatasetFormatInfo]
    issues: list[DatasetIssueCatalogEntry]
    mapping_actions: list[LabeledChoice]
    match_kinds: list[LabeledChoice]
    processing_modes: list[LabeledChoice]
    parents_modes: list[LabeledChoice]
    trust_levels: list[LabeledChoice]
    upload_limits: DatasetUploadLimits
    status_labels: dict[str, str]


__all__ = [name for name in dir() if name[0].isupper() and name not in {'Any', 'Literal'}]
