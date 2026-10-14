"""Region-profile validation.

One function, :func:`validate_profile`, runs on
``POST /region_profiles/validate``, create, clone, ``PUT``,
``POST /region_profiles/test`` (draft) and activate -- see
``src/routers/curation/region_profiles.py``. Async (unlike
``pack_validation.validate_pack``): the detector/OCR/segmenter checks
call out to Triton's repository index and the segmenter's health
endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.config_store.profile_dependency_checks import (
    _check_detector,
    _check_ocr_models,
    _check_segmenter,
    _triton_state,
)
from src.services.config_store.profile_field_checks import (
    _check_candidate_source,
    _check_display,
    _check_name,
    _check_parent_classes,
    _check_ranges,
    _check_text_mode,
    _decode,
    _needs_ocr_models,
)
from src.services.config_store.profile_issues import BYPASSABLE_CODES


if TYPE_CHECKING:
    from src.schemas.curation.config_common import ValidationIssue, ValidationReport
    from src.services.config_store.profile_dependency_checks import (
        RepositoryIndexFn,
        SegmenterHealthFn,
    )


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
    from src.schemas.curation.config_common import ValidationReport
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
    'validate_profile',
]
