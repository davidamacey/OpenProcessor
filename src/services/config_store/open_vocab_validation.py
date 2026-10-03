"""Validation of an ``open_vocab`` prompt set (name, body, and for
activation the segmenter it needs).

Mirrors :mod:`src.services.config_store.profile_validation`: the body is
decoded by the one decoder (:func:`decode_open_vocab_set`), each target
prompt goes through the same text checks a region profile's segmenter
prompt does, and a missing or unreachable segmenter blocks activation only
(``force`` bypasses it). A target named like a primary-detector class is a
warning, never an error: the detector is the cheaper source for it, but the
operator may want SAM 3's boxes anyway.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from src.services.config_store.pack_validation import NAME_RE, RESERVED_NAMES
from src.services.config_store.profile_validation import _issue, check_segmenter_prompt_text
from src.services.detection.open_vocab_set import (
    MAX_ENABLED_TARGETS_CEILING,
    OpenVocabSet,
    decode_open_vocab_set,
)
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.routers.curation._config_common_models import ValidationIssue, ValidationReport

    SegmenterHealthFn = Callable[[], Awaitable[tuple[str, str | None]]]

#: Inclusive ranges; also feeds ``GET /open_vocab/schema`` (one table).
OPEN_VOCAB_FIELD_RANGES: dict[str, tuple[float, float]] = {
    'min_score': (0.0, 1.0),
    'min_area_frac': (0.0, 1.0),
    'max_area_frac': (0.0, 1.0),
    'max_instances': (1, 128),
    'image_max_side': (256, 4096),
    'dedup_iou': (0.0, 1.0),
    'max_enabled_targets': (1, MAX_ENABLED_TARGETS_CEILING),
    'window': (1, 1000),
    'miss_threshold': (1, 1000),
    'sample_floor': (0.0, 1.0),
}

#: Errors ``force: true`` bypasses on activation (an outage that may be
#: stale or racy); everything else is never bypassable.
BYPASSABLE_CODES: frozenset[str] = frozenset({'segmenter_unreachable', 'segmenter_not_configured'})

_MAX_CLASS_NAME_LEN = 64

#: ``ValidationIssue.field`` paths are dotted from the body root, with a list
#: index in brackets: ``targets[2].prompt``, ``gating.tier3_hit_rate.window``,
#: ``image_max_side``. A whole-target issue names ``targets[2]``.
HIT_RATE_PATH = 'gating.tier3_hit_rate'


def target_path(index: int) -> str:
    return f'targets[{index}]'


def _norm(name: str | None) -> str:
    return normalize_class_name(name or '')


def _check_name(name: str | None, existing_names: frozenset[str]) -> list[ValidationIssue]:
    if name is None:
        return []
    if not NAME_RE.match(name):
        return [_issue('open_vocab_name_invalid', 'error', f'{name!r} is not a valid set name')]
    issues = []
    if name in RESERVED_NAMES:
        issues.append(_issue('open_vocab_name_reserved', 'error', f'{name!r} is a reserved name'))
    if name in existing_names:
        issues.append(_issue('name_conflict', 'error', f'{name!r} is already taken'))
    return issues


def _range_issue(path: str, field_name: str, value: float) -> ValidationIssue | None:
    low, high = OPEN_VOCAB_FIELD_RANGES[field_name]
    if low <= value <= high:
        return None
    return _issue(
        'open_vocab_field_range',
        'error',
        f'{path} must be between {low:g} and {high:g}',
        field=path,
    )


def _check_ranges(ov: OpenVocabSet) -> list[ValidationIssue]:
    pairs: list[tuple[str, str, float]] = [
        ('image_max_side', 'image_max_side', ov.image_max_side),
        ('dedup_iou', 'dedup_iou', ov.dedup_iou),
        ('max_enabled_targets', 'max_enabled_targets', ov.max_enabled_targets),
    ]
    hit_rate = ov.gating.tier3_hit_rate
    pairs += [
        (f'{HIT_RATE_PATH}.window', 'window', hit_rate.window),
        (f'{HIT_RATE_PATH}.miss_threshold', 'miss_threshold', hit_rate.miss_threshold),
        (f'{HIT_RATE_PATH}.sample_floor', 'sample_floor', hit_rate.sample_floor),
    ]
    issues: list[ValidationIssue] = []
    for i, t in enumerate(ov.targets):
        base = target_path(i)
        pairs += [
            (f'{base}.min_score', 'min_score', t.min_score),
            (f'{base}.min_area_frac', 'min_area_frac', t.min_area_frac),
            (f'{base}.max_area_frac', 'max_area_frac', t.max_area_frac),
            (f'{base}.max_instances', 'max_instances', t.max_instances),
        ]
        if t.max_area_frac <= 0.0 or t.min_area_frac >= t.max_area_frac:
            issues.append(
                _issue(
                    'open_vocab_field_range',
                    'error',
                    f'{base}: min_area_frac must be below max_area_frac, and max_area_frac above 0',
                    field=f'{base}.max_area_frac',
                )
            )
    if hit_rate.miss_threshold > hit_rate.window:
        issues.append(
            _issue(
                'open_vocab_field_range',
                'error',
                f'{HIT_RATE_PATH}.miss_threshold cannot exceed window',
                field=f'{HIT_RATE_PATH}.miss_threshold',
            )
        )
    issues.extend(i for i in (_range_issue(p, f, v) for p, f, v in pairs) if i is not None)
    return issues


def _check_targets(
    ov: OpenVocabSet, class_names: frozenset[str], detector_class_names: frozenset[str]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    registry = {_norm(n) for n in class_names}
    detector = {_norm(n) for n in detector_class_names}
    seen: set[tuple[str, str]] = set()
    for i, t in enumerate(ov.targets):
        base = target_path(i)
        issues.extend(check_segmenter_prompt_text(t.prompt, sole_leg=True, field=f'{base}.prompt'))
        if t.class_name and (len(t.class_name) > _MAX_CLASS_NAME_LEN or not _norm(t.class_name)):
            issues.append(
                _issue(
                    'open_vocab_class_name_invalid',
                    'error',
                    f'{base}.class_name must contain a letter or digit and be at most '
                    f'{_MAX_CLASS_NAME_LEN} characters',
                    field=f'{base}.class_name',
                )
            )
        key = (t.prompt.strip().casefold(), _norm(t.class_name))
        if key in seen:
            issues.append(
                _issue(
                    'open_vocab_duplicate_target',
                    'error',
                    f'{base} repeats an earlier target (same prompt and class name)',
                    field=base,
                )
            )
        seen.add(key)
        issues.extend(
            _issue(
                'parent_class_unknown',
                'warning',
                f'{base}: parent class {parent!r} is not in the class registry',
                field=f'{base}.parent_classes',
            )
            for parent in t.parent_classes
            if registry and _norm(parent) not in registry
        )
        if not t.class_name:
            continue
        if _norm(t.class_name) in detector:
            issues.append(
                _issue(
                    'open_vocab_detector_class',
                    'warning',
                    f'{base}: {t.class_name!r} is a class the primary detector already '
                    'detects; the detector is the cheaper source',
                    field=f'{base}.class_name',
                )
            )
        if registry and _norm(t.class_name) not in registry:
            issues.append(
                _issue(
                    'open_vocab_class_new',
                    'info',
                    f'{base}: class {_norm(t.class_name)!r} will be added to the class registry '
                    'when the first hit is written',
                    field=f'{base}.class_name',
                )
            )
    return issues


async def _check_segmenter(
    for_activation: bool, segmenter_health: SegmenterHealthFn | None
) -> list[ValidationIssue]:
    severity = 'error' if for_activation else 'warning'
    if not os.environ.get('OP_SEGMENTER_URL', '').strip():
        return [_issue('segmenter_not_configured', severity, 'OP_SEGMENTER_URL is not configured')]
    if segmenter_health is None:
        return []
    try:
        status, last_error = await segmenter_health()
    except Exception as exc:
        status, last_error = 'unavailable', str(exc)
    if status == 'ready':
        return []
    return [_issue('segmenter_unreachable', severity, last_error or 'segmenter is unreachable')]


async def validate_open_vocab(
    name: str | None,
    body: dict[str, Any],
    *,
    existing_names: frozenset[str] = frozenset(),
    for_activation: bool = False,
    segmenter_health: SegmenterHealthFn | None = None,
    class_names: frozenset[str] = frozenset(),
    detector_class_names: frozenset[str] = frozenset(),
    vlm_configured: bool = True,
) -> ValidationReport:
    from src.routers.curation._config_common_models import ValidationReport

    issues = _check_name(name, existing_names)
    try:
        ov = decode_open_vocab_set(name or 'draft', body)
    except ValueError as exc:
        issues.append(_issue('open_vocab_field_invalid', 'error', str(exc)))
    else:
        issues += _check_ranges(ov)
        issues += _check_targets(ov, class_names, detector_class_names)
        enabled = ov.enabled_targets
        if not enabled:
            severity = 'error' if for_activation else 'warning'
            issues.append(_issue('open_vocab_no_enabled_targets', severity, 'no target is enabled'))
        if len(enabled) > ov.max_enabled_targets:
            issues.append(
                _issue(
                    'open_vocab_too_many_targets',
                    'error',
                    f'{len(enabled)} enabled targets exceed max_enabled_targets '
                    f'({ov.max_enabled_targets}); cost is linear in targets',
                    field='targets',
                )
            )
        if ov.gating.tier2_vlm_precheck and not vlm_configured:
            issues.append(
                _issue(
                    'open_vocab_vlm_not_configured',
                    'warning',
                    'tier2_vlm_precheck is on but no vision model is active: the pre-check '
                    'cannot run and every call proceeds',
                    field='gating.tier2_vlm_precheck',
                )
            )
        if enabled:
            issues += await _check_segmenter(for_activation, segmenter_health)
    errors = [i for i in issues if i.severity == 'error']
    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=[i for i in issues if i.severity != 'error'],
        force_allowed=bool(errors) and all(e.code in BYPASSABLE_CODES for e in errors),
    )


__all__ = ['BYPASSABLE_CODES', 'OPEN_VOCAB_FIELD_RANGES', 'validate_open_vocab']
