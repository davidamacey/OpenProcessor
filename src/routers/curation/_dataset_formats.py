"""The served vocabulary of ``GET /datasets/formats``: formats, the issue
catalog, mapping actions and match kinds, processing/parents/trust choices
and the upload limits. The GUI hardcodes none of it."""

from __future__ import annotations

from src.routers.curation._dataset_import_models import (
    DatasetFormatInfo,
    DatasetFormatsResponse,
    DatasetIssueCatalogEntry,
    DatasetUploadLimits,
    LabeledChoice,
)
from src.routers.curation._dataset_import_views import STATUS_LABELS
from src.services.curation.dataset_import import limits
from src.services.curation.dataset_import.issues import ISSUE_CATALOG


def _c(value: str, label: str, description: str = '') -> LabeledChoice:
    return LabeledChoice(value=value, label=label, description=description)


def formats_response() -> DatasetFormatsResponse:
    return DatasetFormatsResponse(
        formats=[
            DatasetFormatInfo(format='auto', label='Detect automatically'),
            DatasetFormatInfo(format='yolo', label='YOLO (data.yaml + images/labels)'),
            DatasetFormatInfo(format='coco', label='COCO (instances JSON)'),
            DatasetFormatInfo(format='openprocessor_export', label='OpenProcessor export'),
        ],
        issues=[
            DatasetIssueCatalogEntry(
                code=code,
                severity=spec.severity,
                blocking=spec.blocking,
                bypassable=spec.bypassable,
                label=spec.label,
            )
            for code, spec in sorted(ISSUE_CATALOG.items())
        ],
        mapping_actions=[
            _c('map', 'Map to a class', 'Use an existing class of this project.'),
            _c('create', 'Create a class', 'Add a new class with this name.'),
            _c('skip', 'Skip', 'Drop these boxes; the frames stay unlabeled.'),
            _c('region', 'Region class', 'Attach these boxes to their parent items.'),
        ],
        match_kinds=[
            _c('same_registry', 'Same class here'),
            _c('exact', 'Exact name'),
            _c('case_insensitive', 'Same name, different case or spacing'),
            _c('merged', 'A class that was merged'),
            _c('synonym', 'Prompt-pack synonym'),
            _c('region', 'The active region class'),
            _c('none', 'No match'),
        ],
        processing_modes=[
            _c('none', 'Import as-is', 'Index the images and import the labels; no detector runs.'),
            _c(
                'propose',
                'Import and propose',
                'Also run the detector to find what the labels missed.',
            ),
        ],
        parents_modes=[
            _c('auto', 'Automatic'),
            _c('labels', 'From the dataset labels'),
            _c('detect', 'Run the detector'),
        ],
        trust_levels=[
            _c('validated', 'Trusted ground truth', 'Imported labels count as validated.'),
            _c('suggestion', 'Suggestions', 'Imported labels are proposals a reviewer confirms.'),
        ],
        upload_limits=DatasetUploadLimits(
            max_bytes=limits.upload_max_bytes(),
            max_files=limits.upload_max_files(),
            ttl_hours=limits.upload_ttl_hours(),
            preview_max_files=limits.preview_max_files(),
        ),
        status_labels=STATUS_LABELS,
    )


__all__ = ['formats_response']
