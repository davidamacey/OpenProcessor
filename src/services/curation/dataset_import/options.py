"""The request-level types of a dataset import, shared by the service layer
and the routes (``extra='forbid'``: a stale or misspelled key is a 422)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.services.curation.dataset_import.mapping import (
    ClassMappingEntry,  # noqa: TC001 - pydantic field type, resolved at runtime
)


Processing = Literal['none', 'propose']
LabelTrust = Literal['validated', 'suggestion']
ParentsMode = Literal['auto', 'labels', 'detect']
MissingLabel = Literal['unlabeled', 'negative']
SourceFormat = Literal['auto', 'yolo', 'coco', 'openprocessor_export']


class CocoAnnotationFileSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    path: str
    images_dir: str
    split: Literal['train', 'val', 'test'] | None = None


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra='forbid')

    path: str
    format: SourceFormat = 'auto'
    coco_annotations: list[CocoAnnotationFileSpec] = Field(default_factory=list)


class DatasetImportOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')

    processing: Processing = 'none'
    label_trust: LabelTrust = 'validated'
    parents: ParentsMode = 'auto'
    freeze_test_split: bool | None = None
    missing_label: MissingLabel = 'unlabeled'
    region_negatives: bool = True
    region_containment: float = Field(default=0.9, ge=0.5, le=1.0)
    name: str = Field(default='dataset_import', min_length=1, max_length=120)
    source_tag: str = Field(default='dataset_import', min_length=1, max_length=120)
    force: bool = False

    def key_fields(self) -> dict[str, object]:
        """The options that change what an import writes (W10.11), hence
        part of ``import_key``. ``name``, ``source_tag`` and ``force`` label
        or gate an import without changing its writes."""
        return {
            'processing': self.processing,
            'label_trust': self.label_trust,
            'parents': self.parents,
            'freeze_test_split': self.freeze_test_split,
            'missing_label': self.missing_label,
            'region_negatives': self.region_negatives,
            'region_containment': self.region_containment,
        }


class DatasetPreviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source: DatasetSource
    mapping: list[ClassMappingEntry] = Field(default_factory=list)
    accept_suggestions: bool = False
    options: DatasetImportOptions = Field(default_factory=DatasetImportOptions)


class DatasetImportRequest(DatasetPreviewRequest):
    expected_import_key: str | None = None


__all__ = [
    'CocoAnnotationFileSpec',
    'DatasetImportOptions',
    'DatasetImportRequest',
    'DatasetPreviewRequest',
    'DatasetSource',
    'LabelTrust',
    'MissingLabel',
    'ParentsMode',
    'Processing',
    'SourceFormat',
]
