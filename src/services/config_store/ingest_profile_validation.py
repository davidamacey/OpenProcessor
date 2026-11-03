"""Validation of an ``ingest_profile`` (name, body shape, and for activation the
Triton models it names).

Shape is checked on every save; servability only when activating (a saved
profile may name a model that is not loaded yet, as a region profile can). The
primary must serve the end2end four-tensor contract ingest reads; the secondary
the single raw tensor the ensemble path reads. ``force`` bypasses only a Triton
outage (``triton_unreachable``, which may be stale); a model that is loaded but
not ready or lacks the outputs is never bypassable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from tritonclient.utils import InferenceServerException

from src.clients.model_adapters import END2END_OUTPUTS
from src.schemas.curation.config_common import ValidationReport
from src.schemas.curation.ingest_profile import IngestProfileBody
from src.services.config_store.pack_validation import NAME_RE, RESERVED_NAMES
from src.services.config_store.profile_issues import issue_builder


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.schemas.curation.config_common import ValidationIssue


#: Errors ``force: true`` bypasses on activation.
BYPASSABLE_CODES: frozenset[str] = frozenset({'triton_unreachable'})
_issue = issue_builder(BYPASSABLE_CODES)


def taken_message(name: str) -> str:
    from src.services.config_store.ingest_profiles import template_names

    if name in template_names():
        return (
            f'{name!r} is the name of a shipped read-only template; '
            f'pick another name or clone it (POST /ingest_profiles/{name}/clone)'
        )
    return f'{name!r} is already taken'


def _check_name(name: str | None, existing_names: frozenset[str]) -> list[ValidationIssue]:
    if name is None:
        return []
    if not NAME_RE.match(name):
        return [_issue('ingest_profile_name_invalid', 'error', f'{name!r} is not a valid name')]
    issues = []
    if name in RESERVED_NAMES:
        issues.append(
            _issue('ingest_profile_name_reserved', 'error', f'{name!r} is a reserved name')
        )
    if name in existing_names:
        issues.append(_issue('name_conflict', 'error', taken_message(name)))
    return issues


def _shape_issues(body: dict[str, Any]) -> list[ValidationIssue]:
    try:
        IngestProfileBody.model_validate(body)
    except ValidationError as exc:
        return [
            _issue(
                'ingest_profile_field_invalid',
                'error',
                f'{".".join(str(p) for p in err["loc"])}: {err["msg"]}',
                field='.'.join(str(p) for p in err['loc']) or None,
            )
            for err in exc.errors()
        ]
    return []


async def _servable_issues(
    pool: Any, model: str, required: Sequence[str], field: str
) -> list[ValidationIssue]:
    try:
        ready = await pool.is_model_ready(model)
        outputs = set(await pool.get_model_output_names(model)) if ready else set()
    except (InferenceServerException, OSError, TimeoutError, RuntimeError) as exc:
        return [
            _issue('triton_unreachable', 'error', f'Triton could not be asked: {exc}', field=field)
        ]
    if not ready:
        return [
            _issue(
                'detector_model_not_ready',
                'error',
                f'{model!r} is not loaded and ready on Triton',
                field=field,
            )
        ]
    missing = [name for name in required if name not in outputs]
    if missing:
        return [
            _issue(
                'detector_model_unusable',
                'error',
                f'{model!r} does not serve the outputs ingest reads: {missing}',
                field=field,
            )
        ]
    return []


async def validate_ingest_profile(
    name: str | None,
    body: dict[str, Any],
    *,
    existing_names: frozenset[str] = frozenset(),
    for_activation: bool = False,
    pool_provider: Any = None,
) -> ValidationReport:
    from src.services.curation.ingest_detect import RAW_OUTPUT

    errors = [*_check_name(name, existing_names), *_shape_issues(body)]
    if for_activation and not errors:
        spec = IngestProfileBody.model_validate(body)
        try:
            pool = pool_provider()
        except RuntimeError as exc:
            errors.append(
                _issue(
                    'triton_unreachable',
                    'error',
                    f'Triton is unavailable: {exc}',
                    field='primary.model',
                )
            )
        else:
            errors.extend(
                await _servable_issues(pool, spec.primary.model, END2END_OUTPUTS, 'primary.model')
            )
            if spec.secondary is not None:
                errors.extend(
                    await _servable_issues(
                        pool, spec.secondary.model, (RAW_OUTPUT,), 'secondary.model'
                    )
                )
    return ValidationReport(
        ok=not errors,
        errors=errors,
        force_allowed=bool(errors) and all(e.code in BYPASSABLE_CODES for e in errors),
    )


__all__ = ['taken_message', 'validate_ingest_profile']
