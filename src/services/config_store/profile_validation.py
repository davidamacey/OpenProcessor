"""Region-profile validation (W4, any_domain_plan.md §4.3).

One function, :func:`validate_profile`, runs on
``POST /region_profiles/validate``, create, clone, ``PUT``,
``POST /region_profiles/test`` (draft) and activate -- see
``src/routers/curation/region_profiles.py``. Async (unlike
``pack_validation.validate_pack``): the detector/OCR/segmenter checks
call out to Triton's repository index and the segmenter's health
endpoint.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.services.config_store.pack_validation import NAME_RE, RESERVED_NAMES
from src.services.detection.region_text import TEXT_READER_MODES
from src.services.labeling.vlm_endpoints import vlm_configured


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.config import DetectionProfile
    from src.routers.curation._config_common_models import ValidationIssue, ValidationReport

    RepositoryIndexFn = Callable[[], Awaitable[list[dict[str, Any]]]]
    SegmenterHealthFn = Callable[[], Awaitable[tuple[str, str | None]]]

#: Inclusive numeric ranges (any_domain_plan.md §4.3); also feeds the
#: ``/region_profiles/schema`` rows' ``min``/``max`` (§7.3) -- one table.
PROFILE_FIELD_RANGES: dict[str, tuple[float, float]] = {
    'confidence_floor': (0.0, 1.0),
    'ocr_det_prob_floor': (0.0, 1.0),
    'text_crop_margin': (0.0, 1.0),
    'batch_limit': (1, 256),
    'input_size': (32, 2048),
    'ocr_det_input_size': (32, 2048),
    'max_regions_per_item': (1, 64),
    # W8.4: (0, 1] -- exclusive 0, checked separately below.
    'region_nms_iou': (0.0, 1.0),
    'gate_hit_window': (1, 1000),
    'gate_hit_miss_threshold': (1, 1000),
    'gate_hit_sample_floor': (0.0, 1.0),
}
_MULTIPLE_OF_32_FIELDS = ('input_size', 'ocr_det_input_size')

#: Errors this validator raises that ``force: true`` bypasses on
#: activation -- unreachable/not-ready dependency checks that are, by
#: their nature, sometimes stale or racy; everything else (field
#: decode/range errors, no_candidate_source, the multi-box pairing
#: error) is never bypassable.
BYPASSABLE_CODES: frozenset[str] = frozenset(
    {
        'detector_model_not_ready',
        'triton_unreachable',
        'segmenter_unreachable',
        'ocr_model_not_ready',
    }
)

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


def _issue(
    code: str,
    severity: str,
    message: str,
    *,
    field: str | None = None,
    detail: dict[str, Any] | None = None,
) -> ValidationIssue:
    from src.routers.curation._config_common_models import ValidationIssue

    return ValidationIssue(
        code=code,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        field=field,
        message=message,
        detail=detail or {},
        bypassable=code in BYPASSABLE_CODES,
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


async def _triton_state(
    get_repository_index: RepositoryIndexFn,
) -> tuple[bool, dict[str, str]]:
    try:
        index = await get_repository_index()
    except Exception:
        return False, {}
    return True, {entry['name']: entry.get('state', '') for entry in index if 'name' in entry}


async def _check_detector(
    profile: DetectionProfile,
    *,
    triton_reachable: bool,
    states: dict[str, str],
    for_activation: bool,
    project_slug: str | None,
) -> list[ValidationIssue]:
    name = profile.detector_model
    if not name:
        return []
    issues: list[ValidationIssue] = []
    from src.services.training.model_classes import (
        is_model_shared,
        model_class_mapping,
        model_owner_project,
    )
    from src.services.training.promoted_models import is_promoted, project_owns_model

    if project_slug is not None and not project_owns_model(name):
        owner = model_owner_project(name)
        if owner is not None:
            if not is_model_shared(name):
                issues.append(
                    _issue(
                        'detector_model_not_shared',
                        'error',
                        f'{name!r} belongs to project {owner!r} and is not shared',
                        field='detector_model',
                        detail={'project': owner},
                    )
                )
            else:
                issues.append(
                    _issue(
                        'detector_model_other_project',
                        'warning',
                        f'{name!r} is shared from project {owner!r}',
                        field='detector_model',
                        detail={'project': owner},
                    )
                )
                try:
                    from src.routers.curation._models_class_mapping import bound_registry

                    mapping = model_class_mapping(name, bound_registry(), project=project_slug)
                    if mapping.unmapped:
                        issues.append(
                            _issue(
                                'detector_model_classes_unmapped',
                                'warning',
                                f"{name!r} has classes unmapped in this project's registry",
                                field='detector_model',
                                detail={'unmapped': list(mapping.unmapped)},
                            )
                        )
                except Exception as exc:  # pragma: no cover - defensive; no registry bound
                    from src.core.logging import get_logger

                    get_logger(__name__).warning(
                        'profile_validation_class_mapping_failed', model=name, error=str(exc)
                    )

    promoted = is_promoted(name)
    if not triton_reachable:
        code = 'triton_unreachable'
        severity = 'error' if for_activation else 'warning'
        issues.append(_issue(code, severity, 'Triton is unreachable', field='detector_model'))
    elif name not in states and not promoted:
        issues.append(
            _issue(
                'detector_model_not_found',
                'error',
                f'{name!r} is not in the Triton repository and not promoted',
                field='detector_model',
                detail={'available': sorted(states)},
            )
        )
    elif name in states and states[name] != 'READY':
        severity = 'error' if for_activation else 'warning'
        issues.append(
            _issue(
                'detector_model_not_ready',
                severity,
                f'{name!r} is not READY (state={states.get(name)!r})',
                field='detector_model',
            )
        )
    return issues


async def _check_ocr_models(
    profile: DetectionProfile,
    *,
    triton_reachable: bool,
    states: dict[str, str],
    for_activation: bool,
) -> list[ValidationIssue]:
    if not _needs_ocr_models(profile):
        return []
    issues: list[ValidationIssue] = []
    if not triton_reachable:
        severity = 'error' if for_activation else 'warning'
        issues.append(
            _issue(
                'triton_unreachable', severity, 'Triton is unreachable', field='ocr_pipeline_model'
            )
        )
        return issues
    for field in ('ocr_pipeline_model', 'ocr_det_model', 'ocr_rec_model'):
        name = getattr(profile, field)
        if not name:
            continue
        if name not in states:
            issues.append(
                _issue(
                    'ocr_model_not_found',
                    'error',
                    f'{name!r} is not in the Triton repository',
                    field=field,
                    detail={'available': sorted(states)},
                )
            )
        elif states[name] != 'READY':
            severity = 'error' if for_activation else 'warning'
            issues.append(
                _issue('ocr_model_not_ready', severity, f'{name!r} is not READY', field=field)
            )
    return issues


def check_segmenter_prompt_text(
    text_prompt: str, *, sole_leg: bool, field: str = 'segmenter_text_prompt'
) -> list[ValidationIssue]:
    """The text-prompt-only checks: served alone by
    ``POST /region_profiles/validate_segmenter_prompt`` and applied to every
    open-vocabulary target prompt (``field`` names where the prompt lives)."""
    if not text_prompt:
        severity = 'error' if sole_leg else 'warning'
        return [_issue('segmenter_prompt_empty', severity, f'{field} is empty', field=field)]
    issues = []
    if len(text_prompt) > 200:
        issues.append(
            _issue('segmenter_prompt_too_long', 'error', f'{field} exceeds 200 chars', field=field)
        )
    if len(text_prompt.split(',')) > 8:
        issues.append(
            _issue(
                'segmenter_prompt_too_many_phrases',
                'error',
                'more than 8 comma-separated phrases',
                field=field,
            )
        )
    if '\n' in text_prompt:
        issues.append(
            _issue(
                'segmenter_prompt_multiline', 'error', f'{field} has multiple lines', field=field
            )
        )
    return issues


async def _check_segmenter(
    profile: DetectionProfile,
    *,
    for_activation: bool,
    segmenter_health: SegmenterHealthFn | None,
) -> list[ValidationIssue]:
    import os

    issues = check_segmenter_prompt_text(
        profile.segmenter_text_prompt, sole_leg=not profile.detector_model
    )
    configured = bool(os.environ.get('OP_SEGMENTER_URL', '').strip())
    if profile.segmenter_text_prompt and not configured:
        issues.append(
            _issue('segmenter_not_configured', 'warning', 'OP_SEGMENTER_URL is not configured')
        )
    elif profile.segmenter_text_prompt and configured and segmenter_health is not None:
        try:
            status, last_error = await segmenter_health()
        except Exception as exc:
            status, last_error = 'unavailable', str(exc)
        if status != 'ready':
            severity = 'error' if for_activation else 'warning'
            issues.append(
                _issue(
                    'segmenter_unreachable',
                    severity,
                    last_error or 'segmenter is unreachable',
                )
            )
    return issues


def _check_candidate_source(profile: DetectionProfile) -> list[ValidationIssue]:
    import os

    configured = bool(os.environ.get('OP_SEGMENTER_URL', '').strip())
    if not profile.detector_model and (not profile.segmenter_text_prompt or not configured):
        return [
            _issue(
                'no_candidate_source',
                'error',
                'no detector and no usable segmenter leg -- this profile can never produce a box',
            )
        ]
    return []


async def validate_profile(
    name: str | None,
    body: dict[str, Any],
    *,
    existing_names: frozenset[str] = frozenset(),
    for_activation: bool = False,
    get_repository_index: RepositoryIndexFn | None = None,
    segmenter_health: SegmenterHealthFn | None = None,
    active_pack: Any | None = None,
    class_names: frozenset[str] | None = None,
    project_slug: str | None = None,
) -> ValidationReport:
    """Validate a region-profile envelope (``{name, body}``)."""
    from src.routers.curation._config_common_models import ValidationReport
    from src.services.triton_control import TritonControlService

    name_issues = _check_name(name, existing_names=existing_names)
    profile, decode_issues = _decode(name, body)
    if profile is None:
        errors = [*name_issues, *decode_issues]
        return ValidationReport(ok=not errors, errors=errors, warnings=[], force_allowed=False)

    all_issues: list[ValidationIssue] = [
        *name_issues,
        *decode_issues,
        *_check_ranges(profile),
        *_check_text_mode(profile),
        *_check_display(profile),
        *_check_parent_classes(profile, class_names=class_names),
        *_check_candidate_source(profile),
    ]

    resolved_get_index = get_repository_index or TritonControlService().get_repository_index
    triton_reachable, states = (
        await _triton_state(resolved_get_index)
        if profile.detector_model or _needs_ocr_models(profile)
        else (True, {})
    )
    all_issues.extend(
        await _check_detector(
            profile,
            triton_reachable=triton_reachable,
            states=states,
            for_activation=for_activation,
            project_slug=project_slug,
        )
    )
    all_issues.extend(
        await _check_ocr_models(
            profile, triton_reachable=triton_reachable, states=states, for_activation=for_activation
        )
    )
    all_issues.extend(
        await _check_segmenter(
            profile, for_activation=for_activation, segmenter_health=segmenter_health
        )
    )

    if for_activation and active_pack is not None:
        from src.services.config_store.pack_validation import validate_pack

        pack_body = active_pack.to_dict()
        pack_report = validate_pack(
            None,
            pack_body,
            profile=profile,
            for_activation=True,
            class_names=class_names,
        )
        all_issues.extend([*pack_report.errors, *pack_report.warnings])

    errors = [i for i in all_issues if i.severity == 'error']
    warnings = [i for i in all_issues if i.severity != 'error']
    force_allowed = bool(errors) and all(e.code in BYPASSABLE_CODES for e in errors)
    return ValidationReport(
        ok=not errors, errors=errors, warnings=warnings, force_allowed=force_allowed
    )


__all__ = [
    'BYPASSABLE_CODES',
    'PROFILE_FIELD_RANGES',
    'validate_profile',
]
