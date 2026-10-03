"""``GET /open_vocab/schema``: one row per set / target / gating field, so a
form renders every control with the right widget and range without a
hardcoded field list. Rows are derived from the decoder's dataclasses; only
the help text and which fields are advanced are declared here."""

from __future__ import annotations

import dataclasses
from typing import Any

from src.routers.curation._open_vocab_models import (
    FieldScope,
    FieldType,
    OpenVocabFieldSchema,
    OpenVocabSchema,
)
from src.services.config_store.open_vocab_validation import OPEN_VOCAB_FIELD_RANGES
from src.services.detection.open_vocab_set import (
    MAX_ENABLED_TARGETS_CEILING,
    GatingConfig,
    HitRateGate,
    OpenVocabSet,
    OpenVocabTarget,
)


_TYPES: dict[str, FieldType] = {
    'str': 'string',
    'int': 'int',
    'float': 'float',
    'bool': 'bool',
    'tuple[str, ...]': 'string_list',
}

#: Fields most sets configure; the rest is tuning.
_CORE = frozenset({'display_name', 'prompt', 'class_name', 'enabled', 'min_score'})

_HELP: dict[str, str] = {
    'prompt': 'What SAM 3 looks for in the whole image, e.g. "traffic cone".',
    'class_name': 'Registry class the hits are stored as. Empty stores them as unlabeled '
    'proposals named by the prompt.',
    'min_score': 'Hits scoring below this are dropped.',
    'min_area_frac': 'Boxes smaller than this fraction of the image are dropped.',
    'max_area_frac': 'Boxes covering more than this fraction of the image are dropped.',
    'max_instances': 'Most boxes kept per image for this target.',
    'parent_classes': 'Only run on images that already hold an item of one of these classes. '
    'Empty runs on every image.',
    'mask': 'Also keep the segmentation outline.',
    'image_max_side': 'Longest side, in pixels, the image is scaled to before the call.',
    'dedup_iou': 'A hit overlapping an existing same-class box this much is not stored again.',
    'run_on_ingest': 'Also run this set on newly ingested images.',
    'max_enabled_targets': 'Most targets that may be enabled together; cost is linear in them.',
    'tier2_vlm_precheck': 'Ask the vision model "is a <prompt> visible?" before spending '
    'a segmenter call.',
    'enabled': 'Include this target in runs.',
    'tier3_hit_rate.enabled': 'Gate by recent hit rate: after a run of misses, sample '
    'instead of running.',
    'window': 'How many recent runs the hit rate is computed over.',
    'miss_threshold': 'Misses within the window that switch the target to sampling.',
    'sample_floor': 'Fraction of images still run while sampling, so recovery is possible.',
}


def _default(f: dataclasses.Field[Any]) -> Any:
    if f.default is not dataclasses.MISSING:
        value = f.default
    elif f.default_factory is not dataclasses.MISSING:
        value = f.default_factory()
    else:
        return None
    return list(value) if isinstance(value, tuple) else value


def _rows(scope: FieldScope, cls: type, skip: frozenset[str]) -> list[OpenVocabFieldSchema]:
    rows = []
    for f in dataclasses.fields(cls):
        if f.name in skip or str(f.type) not in _TYPES:
            continue
        low_high = OPEN_VOCAB_FIELD_RANGES.get(f.name)
        rows.append(
            OpenVocabFieldSchema(
                scope=scope,
                field=f.name,
                label=f.name.replace('_', ' ').capitalize(),
                type=_TYPES[str(f.type)],
                default=_default(f),
                min=low_high[0] if low_high else None,
                max=low_high[1] if low_high else None,
                advanced=f.name not in _CORE,
                help=_HELP.get(f'{scope}.{f.name}', _HELP.get(f.name, '')),
            )
        )
    return rows


def build_open_vocab_schema() -> OpenVocabSchema:
    return OpenVocabSchema(
        fields=[
            *_rows('set', OpenVocabSet, frozenset({'name'})),
            *_rows('target', OpenVocabTarget, frozenset()),
            *_rows('gating', GatingConfig, frozenset()),
            *_rows('tier3_hit_rate', HitRateGate, frozenset()),
        ],
        max_enabled_targets_ceiling=MAX_ENABLED_TARGETS_CEILING,
    )


__all__ = ['build_open_vocab_schema']
