"""VLM endpoint validation (W9.4): field checks, SSRF and external-images
policy, secret references, and the activation-only escalations.

One function, :func:`validate_vlm_endpoint`, runs on validate, create,
clone, PUT, activate and per-run resolution (through
:func:`~src.services.config_store.vlm_gate.enforce_vlm_gate`). Errors block
save and activate; warnings never block. The pack/profile pairing checks
live in :mod:`src.services.config_store.vlm_pairing`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from src.services.labeling.vlm_catalog import catalog_entry
from src.services.labeling.vlm_endpoint_body import FIELD_RANGES, VlmEndpointBody, VlmProbeRecord
from src.services.labeling.vlm_endpoints import (
    ENV_KEY_REF,
    NAME_RE,
    RESERVED_NAMES,
    SECRET_REF_RE,
    api_key_present,
)
from src.services.labeling.vlm_url_policy import (
    Locality,
    UrlSyntaxError,
    acompute_locality,
    aurl_denial,
    parse_endpoint_url,
    sends_images_externally,
)


if TYPE_CHECKING:
    from collections.abc import Collection

    from src.routers.curation._config_common_models import ValidationIssue, ValidationReport

#: Activation errors ``force: true`` may bypass. ``vlm_max_images_exceeds_server``
#: is deliberately absent (every multi-image call would fail), as is every
#: SSRF / external-images / key-reference error.
BYPASSABLE_CODES: frozenset[str] = frozenset(
    {'vlm_not_probed', 'vlm_probe_failed', 'vlm_context_too_small'}
)


def external_policy() -> str:
    """``OP_VLM_EXTERNAL_POLICY``: ``ack`` (default) or ``deny``. Anything
    else fails closed to ``deny``."""
    raw = os.environ.get('OP_VLM_EXTERNAL_POLICY', 'ack').strip().lower() or 'ack'
    return raw if raw == 'ack' else 'deny'


def issue(
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


def build_report(issues: list[ValidationIssue]) -> ValidationReport:
    from src.routers.curation._config_common_models import ValidationReport

    errors = [i for i in issues if i.severity == 'error']
    warnings = [i for i in issues if i.severity != 'error']
    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        force_allowed=bool(errors) and all(e.code in BYPASSABLE_CODES for e in errors),
    )


def check_name(
    name: str | None, *, existing_names: Collection[str] | None
) -> list[ValidationIssue]:
    if name is None:
        return []
    if not NAME_RE.match(name):
        return [
            issue(
                'vlm_name_invalid',
                'error',
                'Names are 2-64 characters of a-z, 0-9, "_", "." or "-", starting with a letter '
                'or digit.',
                field='name',
            )
        ]
    if name in RESERVED_NAMES or name.startswith('_'):
        return [issue('vlm_name_reserved', 'error', f'{name!r} is reserved.', field='name')]
    if existing_names is not None and name in existing_names:
        return [issue('name_conflict', 'error', f'{name!r} already exists.', field='name')]
    return []


def check_ranges(body: VlmEndpointBody) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    if not body.model.strip():
        out.append(issue('vlm_field_range', 'error', 'model must not be empty.', field='model'))
    for field_name, (low, high) in FIELD_RANGES.items():
        value = getattr(body, field_name)
        if value is None:
            continue
        if not low <= value <= high:
            out.append(
                issue(
                    'vlm_field_range',
                    'error',
                    f'{field_name} must be between {low:g} and {high:g}.',
                    field=field_name,
                    detail={'min': low, 'max': high, 'value': value},
                )
            )
    return out


async def check_url(body: VlmEndpointBody) -> tuple[list[ValidationIssue], Locality | None]:
    """Syntax, the never-allowed targets, then locality + the external
    policy. ``locality`` is ``None`` when the URL is unusable."""
    try:
        parse_endpoint_url(body.base_url)
    except UrlSyntaxError as exc:
        return [issue('vlm_url_invalid', 'error', str(exc), field='base_url')], None
    denial = await aurl_denial(body.base_url)
    if denial is not None:
        return [issue(denial.code, 'error', denial.reason, field='base_url')], None
    return [], await acompute_locality(body.base_url)


def check_api_key_ref(
    body: VlmEndpointBody, *, is_env: bool, for_activation: bool
) -> list[ValidationIssue]:
    ref = body.api_key_ref
    if ref is None:
        return []
    valid = bool(SECRET_REF_RE.match(ref)) or (ref == ENV_KEY_REF and is_env)
    if not valid:
        return [
            issue(
                'vlm_api_key_ref_invalid',
                'error',
                'api_key_ref must be secret:<slug> (write it with '
                '`openprocessor vlm key set <slug>`). Environment references are not accepted.',
                field='api_key_ref',
            )
        ]
    # The env built-in's own key is optional (a server with no auth).
    if not api_key_present(ref, is_env_builtin=is_env) and not (is_env and ref == ENV_KEY_REF):
        return [
            issue(
                'vlm_api_key_unresolved',
                'error' if for_activation else 'warning',
                f'{ref} is not present (or is empty) in this deployment.',
                field='api_key_ref',
            )
        ]
    return []


def check_external(
    body: VlmEndpointBody, locality: Locality | None, *, is_env: bool
) -> list[ValidationIssue]:
    if locality is None or not sends_images_externally(locality):
        return []
    out: list[ValidationIssue] = []
    if external_policy() == 'deny':
        out.append(
            issue(
                'vlm_external_denied',
                'error',
                'This deployment does not allow endpoints outside it '
                '(OP_VLM_EXTERNAL_POLICY=deny).',
                field='base_url',
            )
        )
    elif not body.allow_external and not is_env:
        out.append(
            issue(
                'vlm_external_not_acknowledged',
                'error',
                'This endpoint is outside this deployment: crops would be sent to it. '
                'Set allow_external to acknowledge that.',
                field='allow_external',
            )
        )
    return out


def check_probe_for_activation(
    probe: VlmProbeRecord | None, *, is_env: bool
) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    if probe is None:
        if not is_env:
            out.append(
                issue(
                    'vlm_not_probed',
                    'error',
                    'This endpoint has not been tested; run a probe first.',
                )
            )
        return out
    out.extend(
        issue(
            'vlm_max_images_exceeds_server',
            'error',
            item.get('message') or 'The endpoint rejects its own image cap.',
            field='max_images_per_call',
            detail=item.get('detail') or {},
        )
        for item in probe.issues
        if item.get('code') == 'vlm_max_images_exceeds_server'
    )
    if probe.error_codes and not out:
        out.append(
            issue(
                'vlm_probe_failed',
                'error',
                'The last test of this endpoint failed: ' + ', '.join(probe.error_codes) + '.',
                detail={'codes': probe.error_codes},
            )
        )
    return out


async def validate_vlm_endpoint(
    body: VlmEndpointBody,
    *,
    name: str | None,
    probe: VlmProbeRecord | None,
    for_activation: bool,
    existing_names: Collection[str] | None = None,
    is_env: bool = False,
) -> tuple[ValidationReport, Locality | None]:
    """Validate an endpoint body. Returns ``(report, locality)``; ``locality``
    is ``None`` when the URL is unusable. ``existing_names`` (create /
    clone / validate) turns on the ``name_conflict`` check."""
    body = body.normalized()
    issues: list[ValidationIssue] = [
        *check_name(name, existing_names=existing_names),
        *check_ranges(body),
    ]
    url_issues, locality = await check_url(body)
    issues.extend(url_issues)
    issues.extend(check_api_key_ref(body, is_env=is_env, for_activation=for_activation))
    if body.catalog_id is not None and catalog_entry(body.catalog_id) is None:
        issues.append(
            issue(
                'vlm_catalog_id_unknown',
                'warning',
                f'{body.catalog_id!r} is not in the local catalog.',
                field='catalog_id',
            )
        )
    issues.extend(check_external(body, locality, is_env=is_env))
    if for_activation:
        issues.extend(check_probe_for_activation(probe, is_env=is_env))
    return build_report(issues), locality


__all__ = [
    'BYPASSABLE_CODES',
    'build_report',
    'check_name',
    'external_policy',
    'issue',
    'validate_vlm_endpoint',
]
