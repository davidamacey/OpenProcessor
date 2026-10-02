"""``GET /region_profiles/schema``: one row per :class:`DetectionProfile`
field, so a form can render every control with the right widget, range,
group and choice source.

The rows are derived from the dataclass itself (type from the annotation,
default from the field, range from ``PROFILE_FIELD_RANGES``); only the
grouping, the help text and what a field applies to are declared here.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from src.config import DetectionProfile
from src.routers.curation._region_profile_models import (
    AppliesWhen,
    ChoicesFrom,
    FieldType,
    RegionProfileChoice,
    RegionProfileFieldSchema,
    RegionProfileGroup,
    RegionProfileSchema,
)
from src.services.config_store.profile_validation import PROFILE_FIELD_RANGES


_TYPES: dict[str, FieldType] = {
    'str': 'string',
    'int': 'int',
    'float': 'float',
    'bool': 'bool',
    'tuple[int, int, int]': 'rgb',
    'tuple[float, float]': 'float_pair',
    'frozenset[str]': 'string_list',
    'frozenset[int]': 'int_list',
}

GROUPS: tuple[RegionProfileGroup, ...] = (
    RegionProfileGroup(id='identity', label='Name and display'),
    RegionProfileGroup(id='items', label='Which items'),
    RegionProfileGroup(id='detector', label='Detector'),
    RegionProfileGroup(id='segmenter', label='Segmenter'),
    RegionProfileGroup(id='verify', label='Verification'),
    RegionProfileGroup(id='text', label='Text reading'),
    RegionProfileGroup(id='advanced', label='Advanced'),
)

_CHOICES: dict[str, tuple[ChoicesFrom, RegionProfileChoice | None]] = {
    'detector_model': ('detectors', RegionProfileChoice(id='', label='No detector leg')),
    'segmenter_name': ('segmenters', None),
    'ocr_pipeline_model': (
        'ocr_pipeline_models',
        RegionProfileChoice(id='', label='No OCR pipeline'),
    ),
    'ocr_det_model': ('ocr_det_models', RegionProfileChoice(id='', label='No OCR detector')),
    'ocr_rec_model': ('ocr_rec_models', RegionProfileChoice(id='', label='No OCR recognizer')),
    'text_reader': ('text_reader_modes', None),
}

_IDENTITY = ('display_name', 'display_name_singular', 'region_class_name', 'labels_path')
_ITEMS = ('class_ids', 'parent_classes', 'assigns_class')
_VERIFY = ('auto_confirm_aspect', 'auto_confirm_area_frac', 'aspect_min', 'aspect_max')

#: The fields most profiles set; everything else is tuning.
_CORE = frozenset(
    {
        'display_name',
        'display_name_singular',
        'region_class_name',
        'parent_classes',
        'detector_model',
        'confidence_floor',
        'max_regions_per_item',
        'segmenter_text_prompt',
        'text_reader',
    }
)

_HELP: dict[str, str] = {
    'parent_classes': 'Item classes that get the region stage. Empty means every item.',
    'detector_model': 'Triton detector that proposes boxes. Empty turns the detector leg off.',
    'confidence_floor': 'Detections scoring below this are dropped.',
    'max_regions_per_item': 'Boxes kept per item. Above 1 enables multi-box regions.',
    'segmenter_text_prompt': 'Text prompt for the segmenter leg (one short line).',
    'text_reader': 'Which reader fills the region text; none stores no text.',
    'region_class_name': 'Registry class the region boxes are exported as.',
}


def _group(name: str) -> str:
    if name in _IDENTITY:
        return 'identity'
    if name in _ITEMS:
        return 'items'
    if name in _VERIFY:
        return 'verify'
    if name.startswith('segmenter_'):
        return 'segmenter'
    if name.startswith(('text_', 'ocr_')):
        return 'text'
    if name.startswith(('detector_', 'human_detector_')) or name in (
        'input_size',
        'confidence_floor',
        'batch_limit',
        'region_nms_iou',
        'max_regions_per_item',
        'letterbox_fill',
        'feature_output',
        'secondary_shape_groups',
    ):
        return 'detector'
    return 'advanced'


def _applies_when(name: str) -> AppliesWhen | None:
    if name.startswith('text_hint_'):
        return 'text_hint'
    if name.startswith(('text_', 'ocr_')) and name != 'text_reader':
        return 'reads_text'
    if name.startswith('segmenter_'):
        return 'segmenter'
    if name.startswith('detector_') or name in ('input_size', 'confidence_floor', 'batch_limit'):
        return 'detector'
    return None


def _default(f: dataclasses.Field[Any]) -> Any:
    if f.default is not dataclasses.MISSING:
        value = f.default
    elif f.default_factory is not dataclasses.MISSING:
        value = f.default_factory()
    else:
        return None
    if isinstance(value, frozenset):
        return sorted(value)
    return list(value) if isinstance(value, tuple) else value


def build_region_profile_schema() -> RegionProfileSchema:
    rows: list[RegionProfileFieldSchema] = []
    for f in dataclasses.fields(DetectionProfile):
        if f.name == 'name':
            continue
        choices_from, empty_choice = _CHOICES.get(f.name, (None, None))
        low_high = PROFILE_FIELD_RANGES.get(f.name)
        rows.append(
            RegionProfileFieldSchema(
                field=f.name,
                label=f.name.replace('_', ' ').capitalize(),
                group=_group(f.name),
                type=_TYPES[str(f.type)],
                default=_default(f),
                min=low_high[0] if low_high else None,
                max=low_high[1] if low_high else None,
                advanced=f.name not in _CORE,
                applies_when=_applies_when(f.name),
                choices_from=choices_from,
                empty_choice=empty_choice,
                help=_HELP.get(f.name, ''),
            )
        )
    return RegionProfileSchema(fields=rows, groups=list(GROUPS))


__all__ = ['GROUPS', 'build_region_profile_schema']
