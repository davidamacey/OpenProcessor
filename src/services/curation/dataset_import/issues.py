"""Dataset-import issue vocabulary (W10.4): a stable, served catalog of
codes, so ``GET /datasets/formats`` never needs a hardcoded copy on the
client.

Not the full any_domain_plan.md W10.4 table (deferred codes are the
OpenProcessor-export-specific ones — ``op_export.py`` is out of scope
this pass, see the wave report) — every code this pass's readers
(``yolo.py``, ``coco.py``) and mapping (``mapping.py``) actually emit is
here, plus the DatasetIssue/DatasetIssueSample shapes ``scan.py`` and the
routes build responses from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Severity = Literal['error', 'warning', 'info']

DatasetIssueCode = Literal[
    'dataset_path_not_allowed',
    'format_undetected',
    'dataset_too_large',
    'data_yaml_invalid',
    'data_yaml_names_missing',
    'split_dir_missing',
    'coco_json_invalid',
    'coco_category_duplicate_name',
    'label_file_missing',
    'label_file_orphan',
    'label_row_malformed',
    'label_class_out_of_range',
    'label_coords_out_of_range',
    'label_coords_clamped',
    'label_box_degenerate',
    'yolo_polygon_to_box',
    'coco_crowd_skipped',
    'coco_bbox_out_of_image',
    'image_size_mismatch',
    'image_path_not_servable',
    'image_duplicate_in_dataset',
    'images_already_indexed',
    'already_imported',
    'class_unmapped',
    'class_mapping_conflict',
    'class_name_exists',
    'class_mapped_to_deprecated',
    'class_index_name_mismatch',
    'region_profile_required',
    'region_class_name_differs',
    'region_box_no_parent',
    'label_conflicts_locked',
]


@dataclass(frozen=True)
class IssueSpec:
    severity: Severity
    blocking: bool
    bypassable: bool
    label: str


# Served by GET /datasets/formats so the GUI hardcodes none of it.
ISSUE_CATALOG: dict[str, IssueSpec] = {
    'dataset_path_not_allowed': IssueSpec('error', True, False, 'Dataset path is not allowed'),
    'format_undetected': IssueSpec('error', True, False, 'Could not detect a dataset format'),
    'dataset_too_large': IssueSpec('error', True, False, 'Dataset exceeds the preview file limit'),
    'data_yaml_invalid': IssueSpec('error', True, False, 'data.yaml is invalid'),
    'data_yaml_names_missing': IssueSpec('error', True, False, 'data.yaml has no class names'),
    'split_dir_missing': IssueSpec('warning', False, False, 'A declared split resolves to nothing'),
    'coco_json_invalid': IssueSpec('error', True, False, 'COCO annotation file is invalid'),
    'coco_category_duplicate_name': IssueSpec(
        'error', True, False, 'Two COCO categories share one name'
    ),
    'label_file_missing': IssueSpec('warning', False, False, 'Image has no label file'),
    'label_file_orphan': IssueSpec('info', False, False, 'Label file has no matching image'),
    'label_row_malformed': IssueSpec('warning', False, False, 'Label row is malformed'),
    'label_class_out_of_range': IssueSpec(
        'warning', False, False, 'Label row class id out of range'
    ),
    'label_coords_out_of_range': IssueSpec(
        'warning', False, False, 'Label row coordinates out of range'
    ),
    'label_coords_clamped': IssueSpec(
        'info', False, False, 'Label row coordinates clamped to [0,1]'
    ),
    'label_box_degenerate': IssueSpec('warning', False, False, 'Label row box has zero area'),
    'yolo_polygon_to_box': IssueSpec('info', False, False, 'Polygon/OBB row imported as its box'),
    'coco_crowd_skipped': IssueSpec('info', False, False, 'COCO crowd annotation skipped'),
    'coco_bbox_out_of_image': IssueSpec('warning', False, False, 'COCO bbox outside the image'),
    'image_size_mismatch': IssueSpec(
        'warning', False, False, 'COCO image size does not match the decoded image'
    ),
    'image_path_not_servable': IssueSpec('warning', False, False, 'Image path is not servable'),
    'image_duplicate_in_dataset': IssueSpec(
        'info', False, False, 'Two dataset files share the same image bytes'
    ),
    'images_already_indexed': IssueSpec(
        'info', False, False, 'Image already indexed in this project'
    ),
    'already_imported': IssueSpec('info', False, False, 'This import key was already completed'),
    'class_unmapped': IssueSpec('error', True, False, 'A dataset class with boxes has no mapping'),
    'class_mapping_conflict': IssueSpec(
        'error', True, False, 'Two "create" actions normalize to one name'
    ),
    'class_name_exists': IssueSpec('error', True, False, 'A "create" name already exists'),
    'class_mapped_to_deprecated': IssueSpec('error', True, False, 'Mapped to a deprecated class'),
    'class_index_name_mismatch': IssueSpec(
        'info', False, False, 'The old index-based mapping would have mislabeled this class'
    ),
    'region_profile_required': IssueSpec(
        'error', True, False, 'A region mapping needs an active region profile'
    ),
    'region_class_name_differs': IssueSpec(
        'warning', False, False, "Region class name differs from the profile's region class name"
    ),
    'region_box_no_parent': IssueSpec(
        'info', False, False, 'Boxes with no qualifying parent become standalone items'
    ),
    'label_conflicts_locked': IssueSpec(
        'warning', False, False, 'Label conflicted with a locked (human/import) item'
    ),
}


@dataclass
class DatasetIssueSample:
    file: str
    line: int | None = None
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class DatasetIssue:
    code: str
    severity: Severity
    blocking: bool
    bypassable: bool
    message: str
    count: int = 0
    samples: list[DatasetIssueSample] = field(default_factory=list)


_MAX_SAMPLES = 20


class IssueCollector:
    """Aggregates issues by code with exact counts and up to
    :data:`_MAX_SAMPLES` samples, in first-seen (scan) order."""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._counts: dict[str, int] = {}
        self._samples: dict[str, list[DatasetIssueSample]] = {}

    def add(
        self,
        code: str,
        *,
        file: str,
        line: int | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        if code not in self._counts:
            self._order.append(code)
            self._counts[code] = 0
            self._samples[code] = []
        self._counts[code] += 1
        if len(self._samples[code]) < _MAX_SAMPLES:
            self._samples[code].append(
                DatasetIssueSample(file=file, line=line, detail=detail or {})
            )

    def issues(self) -> list[DatasetIssue]:
        out: list[DatasetIssue] = []
        for code in self._order:
            spec = ISSUE_CATALOG.get(code, IssueSpec('info', False, False, code))
            out.append(
                DatasetIssue(
                    code=code,
                    severity=spec.severity,
                    blocking=spec.blocking,
                    bypassable=spec.bypassable,
                    message=spec.label,
                    count=self._counts[code],
                    samples=list(self._samples[code]),
                )
            )
        return out

    def has_unbypassable_blocking(self, *, force: bool) -> bool:
        """Whether ``POST /datasets/imports`` must refuse (W10.4): any
        blocking issue present, unless every blocking issue is
        bypassable and ``force`` is set."""
        blocking_codes = [
            c
            for c in self._order
            if ISSUE_CATALOG.get(c, IssueSpec('info', False, False, c)).blocking
        ]
        if not blocking_codes:
            return False
        return not (
            force and all(ISSUE_CATALOG[c].bypassable for c in blocking_codes if c in ISSUE_CATALOG)
        )


__all__ = [
    'ISSUE_CATALOG',
    'DatasetIssue',
    'DatasetIssueCode',
    'DatasetIssueSample',
    'IssueCollector',
    'IssueSpec',
    'Severity',
]
