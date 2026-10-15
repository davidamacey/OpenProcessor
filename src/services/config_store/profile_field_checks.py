"""Region-profile checks that need no network: name, field decode and ranges, text
reading mode, display names, parent classes and the candidate source."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.config.env import env_str
from src.services.config_store.pack_validation import NAME_RE, RESERVED_NAMES
from src.services.config_store.profile_issues import _issue
from src.services.detection.region_text import TEXT_READER_MODES
from src.services.labeling.vlm_endpoints import vlm_configured


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.config import DetectionProfile
    from src.schemas.curation.config_common import ValidationIssue

    RepositoryIndexFn = Callable[[], Awaitable[list[dict[str, Any]]]]
    SegmenterHealthFn = Callable[[], Awaitable[tuple[str, str | None]]]


#: Inclusive numeric ranges (any_domain_plan.md §4.3); also feeds the
#: ``/region_profiles/schema`` rows' ``min``/``max`` (§7.3) -- one table.
PROFILE_FIELD_RANGES: dict[str, tuple[float, float]] = {
    'confidence_floor': (0.0, 1.0),
    'text_crop_margin': (0.0, 1.0),
    'batch_limit': (1, 256),
    'input_size': (32, 2048),
    'max_regions_per_item': (1, 64),
    # (0, 1] -- exclusive 0, checked separately below.
    'region_nms_iou': (0.0, 1.0),
    'gate_hit_window': (1, 1000),
    'gate_hit_miss_threshold': (1, 1000),
    'gate_hit_sample_floor': (0.0, 1.0),
}
_MULTIPLE_OF_32_FIELDS = ('input_size',)

_TEXT_READING_MODES = frozenset({'vlm', 'vlm_then_ocr', 'both'})
_OCR_NEEDING_MODES = frozenset({'ocr', 'vlm_then_ocr', 'both'})
_DEFAULT_TEXT_FIELDS = (
    'text_crop_margin',
    'text_crop_min_height',
    'text_min_height_ratio',
    'text_border_margin',
    'text_uppercase',
    'text_charset',
    'text_join',
    'text_len_min',
    'text_len_max',
    'text_stopwords',
    'text_min_confidence',
    'text_format',
    'text_placeholders',
    'text_reject_sequences',
)


def _check_name(name: str | None, *, existing_names: frozenset[str]) -> list[ValidationIssue]:
    if name is None:
        return []
    if not NAME_RE.match(name):
        return [_issue('profile_name_invalid', 'error', f'{name!r} is not a valid profile name')]
    issues = []
    if name in RESERVED_NAMES:
        issues.append(_issue('profile_name_reserved', 'error', f'{name!r} is a reserved name'))
    if name in existing_names:
        issues.append(_issue('name_conflict', 'error', f'{name!r} is already taken'))
    return issues


def _decode(
    name: str | None, body: dict[str, Any]
) -> tuple[DetectionProfile | None, list[ValidationIssue]]:
    from src.services.detection.profile_registry import region_profile_from_dict

    try:
        profile = region_profile_from_dict({**body, 'name': name or 'draft'}, source='validate')
    except ValueError as exc:
        msg = str(exc)
        code = 'profile_field_unknown' if 'unknown field' in msg else 'profile_field_type'
        return None, [_issue(code, 'error', msg)]
    if profile.region_class_name and not NAME_RE.match(profile.region_class_name):
        return profile, [
            _issue(
                'region_class_name_invalid',
                'error',
                f'{profile.region_class_name!r} is not a valid region class name',
                field='region_class_name',
            )
        ]
    return profile, []


def _check_ranges(profile: DetectionProfile) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field, (lo, hi) in PROFILE_FIELD_RANGES.items():
        value = getattr(profile, field)
        low = lo
        if field == 'region_nms_iou' and value <= 0:
            issues.append(
                _issue(
                    'profile_field_range',
                    'error',
                    f'{field} must be in (0, {hi}]',
                    field=field,
                    detail={'min': lo, 'max': hi, 'value': value},
                )
            )
            continue
        if not (low <= value <= hi):
            issues.append(
                _issue(
                    'profile_field_range',
                    'error',
                    f'{field} must be between {lo} and {hi}',
                    field=field,
                    detail={'min': lo, 'max': hi, 'value': value},
                )
            )
    issues.extend(
        _issue(
            'profile_field_range',
            'error',
            f'{field} must be a multiple of 32',
            field=field,
            detail={'value': getattr(profile, field)},
        )
        for field in _MULTIPLE_OF_32_FIELDS
        if getattr(profile, field) % 32 != 0
    )
    if profile.gate_hit_miss_threshold > profile.gate_hit_window:
        issues.append(
            _issue(
                'profile_field_range',
                'error',
                'gate_hit_miss_threshold cannot exceed gate_hit_window',
                field='gate_hit_miss_threshold',
                detail={
                    'value': profile.gate_hit_miss_threshold,
                    'window': profile.gate_hit_window,
                },
            )
        )
    lo_frac, hi_frac = profile.auto_confirm_area_frac
    if not (0 <= lo_frac < hi_frac <= 1):
        issues.append(
            _issue(
                'profile_field_range',
                'error',
                'auto_confirm_area_frac must satisfy 0 <= min < max <= 1',
                field='auto_confirm_area_frac',
                detail={'value': list(profile.auto_confirm_area_frac)},
            )
        )
    fill = profile.letterbox_fill
    if len(fill) != 3 or not all(isinstance(c, int) and 0 <= c <= 255 for c in fill):
        issues.append(
            _issue(
                'profile_field_range',
                'error',
                'letterbox_fill must be 3 integers in 0..255',
                field='letterbox_fill',
                detail={'value': list(fill)},
            )
        )
    return issues


def _check_text_mode(profile: DetectionProfile) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if profile.text_reader not in TEXT_READER_MODES:
        issues.append(
            _issue(
                'text_reader_invalid',
                'error',
                f'{profile.text_reader!r} is not one of {sorted(TEXT_READER_MODES)}',
                field='text_reader',
            )
        )
        return issues
    for field in ('text_charset', 'text_format', 'text_pattern'):
        value = getattr(profile, field)
        if not value:
            continue
        try:
            re.compile(value)
        except re.error as exc:
            issues.append(_issue('text_regex_invalid', 'error', f'{field}: {exc}', field=field))
    if profile.text_reader == 'none':
        non_default = [
            f
            for f in _DEFAULT_TEXT_FIELDS
            if getattr(profile, f) != getattr(type(profile)(name='_ref'), f)
        ]
        if non_default:
            issues.append(
                _issue(
                    'text_fields_ignored',
                    'warning',
                    'text_reader is none; these fields have no effect',
                    detail={'fields': non_default},
                )
            )
        return issues
    if profile.text_reader in _TEXT_READING_MODES and not vlm_configured():
        issues.append(
            _issue('vlm_not_configured', 'warning', 'no VLM is configured; falls back to OCR')
        )
    if profile.text_hint_enabled and not profile.ocr_pipeline_model:
        issues.append(
            _issue(
                'text_hint_inactive_no_ocr',
                'warning',
                'text_hint_enabled but ocr_pipeline_model is empty',
                field='ocr_pipeline_model',
            )
        )
    return issues


def _needs_ocr_models(profile: DetectionProfile) -> bool:
    if profile.text_reader == 'none':
        return False
    return profile.text_reader in _OCR_NEEDING_MODES or (
        profile.text_reader == 'vlm' and not vlm_configured()
    )


def _check_display(profile: DetectionProfile) -> list[ValidationIssue]:
    issues = []
    if not profile.display_name or not profile.display_name_singular:
        issues.append(
            _issue(
                'display_name_missing',
                'warning',
                "display_name / display_name_singular empty; the UI falls back to 'Regions'/'Region'",
            )
        )
    return issues


def _check_parent_classes(
    profile: DetectionProfile, *, class_names: frozenset[str] | None
) -> list[ValidationIssue]:
    if class_names is None:
        return []
    return [
        _issue(
            'parent_class_unknown',
            'warning',
            f'parent class {name!r} is not in the registry (allowed for a proposal name)',
            field='parent_classes',
            detail={'class_name': name},
        )
        for name in sorted(profile.parent_classes)
        if name not in class_names
    ]


def _check_candidate_source(profile: DetectionProfile) -> list[ValidationIssue]:

    configured = bool(env_str('OP_SEGMENTER_URL'))
    if not profile.detector_model and (not profile.segmenter_text_prompt or not configured):
        return [
            _issue(
                'no_candidate_source',
                'error',
                'no detector and no usable segmenter leg -- this profile can never produce a box',
            )
        ]
    return []
