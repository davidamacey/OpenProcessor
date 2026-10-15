"""Profiles, results and artifact names shared by the dataset exporters.

:class:`~src.services.curation.export.GenericYoloExportService` (the
multi-class exporter) and
:class:`~src.services.curation.export_single_class.SingleClassExportService`
(the class-subset exporter) write the same artifact set next to a dataset,
so the filenames and the per-split counter live here once. Each exporter's
own knobs (``ExportProfile`` / ``SingleClassExportProfile``) and result
envelope sit beside them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.curation.export_split import SplitMode


BoxSource = Literal['item', 'region']

# Artifacts written next to every dataset. `GET /curation/export/registry/{artifact}`
# serves ARTIFACT_FILENAMES read-only: a fixed, whitelisted set, never
# derived from user input.
MANIFEST_FILENAME = 'manifest.json'
DATA_YAML_FILENAME = 'data.yaml'
CLASS_REGISTRY_FILENAME = 'class_registry.json'
LABEL_STATS_FILENAME = 'label_stats.json'
STRATUM_MAP_FILENAME = 'stratum_map.json'

ARTIFACT_FILENAMES: dict[str, str] = {
    'manifest': MANIFEST_FILENAME,
    'data_yaml': DATA_YAML_FILENAME,
    'class_registry': CLASS_REGISTRY_FILENAME,
    'label_stats': LABEL_STATS_FILENAME,
}

REGISTRY_ARTIFACT_CONTENT_TYPES: dict[str, str] = {
    'class_registry.json': 'application/json',
    'data.yaml': 'application/x-yaml',
    'manifest.json': 'application/json',
    'label_stats.json': 'application/json',
}


@dataclass(frozen=True)
class ExportProfile:
    """Deployment-tunable export knobs."""

    name: str = 'default'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio is whatever remains: 1 - train_ratio - val_ratio.


@dataclass
class SplitCounts:
    train: int = 0
    val: int = 0
    test: int = 0

    def to_dict(self) -> dict[str, int]:
        return {'train': self.train, 'val': self.val, 'test': self.test}


@dataclass
class ExportResult:
    """What one export wrote. ``split_counts`` / ``image_count`` count
    images; ``split_object_counts`` / ``object_count`` count label lines."""

    export_dir: str
    version_tag: str
    manifest_path: str
    data_yaml_path: str
    dataset_sha: str
    split_counts: SplitCounts
    image_count: int
    class_count: int
    classes_with_objects: int
    started_at: str
    finished_at: str
    current_symlink: str
    object_count: int = 0
    split_object_counts: SplitCounts = field(default_factory=SplitCounts)
    unlabeled_items_on_exported_images: int = 0
    images_with_unlabeled_items: int = 0
    images_dropped_not_fully_labeled: int = 0
    skipped_items: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SingleClassExportProfile:
    """Deployment-supplied configuration for one narrowed export.

    This is the whole reason the exporter is generic: a deployment names
    its own target classes, its own region vocabulary and its own output
    location here, instead of the exporter hardcoding any of them.
    """

    # Directory name under the export root, and the default dataset
    # identity recorded in the manifest. Keeps a narrowed export's
    # artifacts (and its own `current` symlink) out of the multi-class
    # export root, so the two can never clobber each other.
    name: str = 'single_class'

    # Ordered registry class ids forming this export's vocabulary. One id
    # is the single-class case; several is a class-subset export. The
    # dense label id written into the .txt files is the INDEX INTO THIS
    # TUPLE, so the ordering here is the dataset's class order and is
    # frozen into class_registry.json's export_id_map.
    #
    # Required for box_source='item'. For box_source='region' it is
    # instead an optional filter on which parent items' regions are
    # collected (empty = every item's region).
    class_ids: tuple[int, ...] = ()

    # Where each label box comes from: the item's own `bbox_norm`, or the
    # item's region-of-interest sub-annotation (`RegionFields.bbox_norm`).
    box_source: BoxSource = 'item'

    # Class name for the region vocabulary in region mode, where the
    # single exported class is the region itself rather than a registry
    # entry. Purely a label in data.yaml.
    region_class_name: str = 'region'

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio is whatever remains: 1 - train_ratio - val_ratio.

    # Absolute output root override. None = `CurationConfig.export_root / name`.
    output_root: Path | None = None
    current_link_name: str = 'current'

    # Recorded in the manifest so a consumer can tell a narrowed export
    # from a full multi-class one without inspecting data.yaml. Read by
    # `src/services/training/preflight_checks.py`'s single-class preflight branch.
    dataset_kind: str = 'single_class'

    # 'keep_imported' honours the split a dataset import filed each frame
    # under (``dataset_split``); 'recompute' ignores it.
    split_mode: SplitMode = 'keep_imported'


@dataclass
class SingleClassExportResult:
    export_dir: str
    version_tag: str
    manifest_path: str
    data_yaml_path: str
    dataset_sha: str
    frozen_test_sha: str
    split_counts: SplitCounts
    image_count: int
    class_count: int
    positive_images: int
    background_images: int
    started_at: str
    finished_at: str
    current_symlink: str
