"""Region-profile checks that call out: the Triton repository index (detector and OCR
models) and the segmenter prompt and health."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.env import env_str
from src.services.config_store.profile_field_checks import _needs_ocr_models
from src.services.config_store.profile_issues import _issue


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.config import DetectionProfile
    from src.schemas.curation.config_common import ValidationIssue

    RepositoryIndexFn = Callable[[], Awaitable[list[dict[str, Any]]]]
    SegmenterHealthFn = Callable[[], Awaitable[tuple[str, str | None]]]


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
                    from src.clients.curation_opensearch.registry import get_class_registry

                    mapping = model_class_mapping(
                        name, get_class_registry().load(), project=project_slug
                    )
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
    for field in ('ocr_pipeline_model', 'ocr_rec_model'):
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

    issues = check_segmenter_prompt_text(
        profile.segmenter_text_prompt, sole_leg=not profile.detector_model
    )
    configured = bool(env_str('OP_SEGMENTER_URL'))
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
