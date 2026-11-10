"""Request models for the YOLO and single-class exports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Runtime import: pydantic resolves the Literal annotation from module globals.
from src.services.curation.item_filter import ItemFilter


class ExportYoloRequest(BaseModel):
    # G-16: this endpoint used to silently ignore unknown keys (e.g. a
    # 'classes' field a caller expected to narrow the export, which
    # actually lives on ExportSingleClassRequest / train include_classes).
    # 422 on typos/misplaced fields instead of quietly exporting everything.
    model_config = ConfigDict(extra='forbid')

    export_dir: str | None = None
    # Narrow the exported items with the shared item filter (class names,
    # confidence band, box size, origin, ...); recorded in manifest.json.
    item_filter: ItemFilter | None = None
    # Free-form version tag (e.g. 'v7.0a'). Recorded in manifest.json only.
    version_tag: str = ''
    # RNG seed for the stratified split. Recording it in the manifest is what
    # makes an export re-derivable.
    seed: int = 42
    # Cap on exported source images (a class-balanced sample; every object
    # on a kept image stays). None = full export.
    max_images: int | None = None
    # Keep one image per near-duplicate group of the project's current run
    # (a burst collapses to one image); 409 when the project has no run yet.
    collapse_near_duplicates: bool = False
    # False (default): an image that also holds unreviewed objects (or
    # objects on a class the export leaves out) is exported with its
    # validated objects labeled; the manifest counts the rest, and training
    # preflight warns, because a detector learns unlabeled objects as
    # background. True: leave such images out.
    require_fully_labeled_images: bool = False
    # 'keep_imported' (default) keeps the split a dataset import filed each
    # frame under; 'recompute' ignores it and splits by the exporter's rule.
    split_mode: Literal['keep_imported', 'recompute'] = 'keep_imported'
    # Write an imported reviewed-negative frame as an empty label file when
    # its negative_for covers every class this export has objects for.
    include_negative_frames: bool = True


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
    # Keep one frame per near-duplicate group of the project's current run
    # (a burst collapses to one frame); 409 when the project has no run yet.
    collapse_near_duplicates: bool = False
    # 'whole_frame' (full source frame) or 'item_crop' (parent item crop
    # with the region re-projected) — for whole-image vs crop training
    # A/B. 'item_crop' requires box_source='region'.
    image_mode: Literal['whole_frame', 'item_crop'] = 'whole_frame'
    # Longest output side in px: 640 for rapid iteration, 1280 for the full run.
    img_max_side: int = 1280
    # False writes labels + artifacts only (fast dry run, no pixel copy).
    copy_images: bool = True
    # 'keep_imported' (default) keeps the split a dataset import filed each
    # frame under; 'recompute' ignores it and splits by the exporter's rule.
    split_mode: Literal['keep_imported', 'recompute'] = 'keep_imported'


class SingleClassSplitCounts(BaseModel):
    train: int
    val: int
    test: int


class ExportSingleClassResponse(BaseModel):
    """``POST /export/single_class``: the finished export's integrity envelope."""

    status: Literal['success']
    export_dir: str
    version_tag: str
    manifest_path: str
    data_yaml_path: str
    dataset_sha: str
    frozen_test_sha: str
    split_counts: SingleClassSplitCounts
    image_count: int
    class_count: int
    positive_images: int
    background_images: int
    positives_zero_warning: bool = Field(
        description='True when the dataset has no positives and cannot train a detector.'
    )
    current_symlink: str
    started_at: str | None
    finished_at: str | None


class ExportSingleClassStatus(BaseModel):
    """``GET /export/single_class/status``.

    ``idle`` carries only ``status``, ``last_run`` and ``profile_name``; ``unknown`` adds
    ``export_dir``; ``success`` adds the manifest fields (``null`` when the manifest lacks
    one). Served with ``response_model_exclude_unset`` so a branch never gains keys.
    """

    status: Literal['idle', 'unknown', 'success']
    last_run: str | None = None
    profile_name: str
    export_dir: str | None = None
    dataset_kind: str | None = None
    dataset_sha: str | None = None
    frozen_test_sha: str | None = None
    class_count: int | None = None
    class_names: list[str] | None = None
    image_count: int | None = None
    positive_images: int | None = None
    background_images: int | None = None
    false_positive_background_images: int | None = None
    positives_zero_warning: bool | None = None
    split_counts: dict[str, Any] | None = None
