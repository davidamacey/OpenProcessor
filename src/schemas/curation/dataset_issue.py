"""Wire shape and code vocabulary of one dataset-import issue, shared by the
preview, the job, the issue-page route and the structured 4xx detail
(``ConfigErrorDetail.issues``). Its own module so the error-detail model
can import it without a cycle."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DatasetIssueCode = Literal[
    'dataset_path_not_allowed',
    'format_undetected',
    'dataset_too_large',
    'dataset_file_too_large',
    'data_yaml_invalid',
    'data_yaml_names_missing',
    'data_yaml_names_sparse',
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
    'manifest_unreadable',
    'names_mismatch',
    'test_split_changed',
    'op_export_detected',
    'dataset_sha_mismatch',
    'test_frozen_from_manifest',
    'test_frozen_missing',
    'stratum_map_partial',
    'item_crop_export_imported_as_frames',
    'images_reused_by_stem',
    'image_unreadable',
]


class DatasetIssueSampleWire(BaseModel):
    file: str
    line: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class DatasetIssueWire(BaseModel):
    id: str = Field(
        default='',
        description='Unique within the response (`code[:subject]`, `#2` on a repeat).',
    )
    code: DatasetIssueCode
    severity: Literal['error', 'warning', 'info']
    blocking: bool
    bypassable: bool
    message: str
    count: int
    samples: list[DatasetIssueSampleWire] = Field(default_factory=list)


__all__ = ['DatasetIssueCode', 'DatasetIssueSampleWire', 'DatasetIssueWire']
