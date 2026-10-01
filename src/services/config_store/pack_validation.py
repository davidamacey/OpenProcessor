"""Prompt-pack validation (W3, any_domain_plan.md §3.3).

One function, :func:`validate_pack`, runs on ``POST /prompt_packs/validate``,
create, clone (on the resulting body), ``PUT``, ``POST /prompt_packs/test``
(draft) and activate -- see ``src/routers/curation/prompt_packs.py``.
"""

from __future__ import annotations

import re
import string
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Any

from src.services.labeling.region_overlay import REPLY_TEXT_KEY
from src.services.labeling.vlm_prompts import FORMATTED_PLACEHOLDERS, REPLY_KEY_CONTRACT, PromptPack


if TYPE_CHECKING:
    # Deferred at runtime (imported inside functions below): this module is
    # imported by src.routers.curation.prompt_packs, which is itself
    # imported (as a side effect) by src.routers.curation.__init__ --
    # importing _config_common_models at module scope here would re-enter
    # that package's __init__ while it is still executing (same fix
    # keymap_validator.py already applies for the same reason).
    from src.config.detection_profile import DetectionProfile
    from src.routers.curation._config_common_models import ValidationIssue, ValidationReport


#: Slug rule shared by packs and profiles (any_domain_plan.md §3.3).
NAME_RE = re.compile(r'^[a-z0-9][a-z0-9_.-]{1,63}$')
RESERVED_NAMES: frozenset[str] = frozenset({'off', 'none', 'default'})

MAX_FIELD_LEN = 20_000
MAX_MAP_ENTRIES = 500

_STRING_FIELDS = tuple(
    f.name
    for f in dc_fields(PromptPack)
    if f.name not in ('name', 'class_descriptions', 'synonyms')
)
_MAP_FIELDS = ('class_descriptions', 'synonyms')

#: Errors this validator raises that ``force: true`` can bypass on
#: activation. Empty: the pack side's only activation-only error
#: (``pack_multi_region_keys_missing``, raised by the profile-pairing
#: check too) is never bypassable (any_domain_plan.md §3.3) -- a
#: bypassed multi-box mismatch would silently drop verdicts.
BYPASSABLE_CODES: frozenset[str] = frozenset()

_TEXT_ASKING_KEYS = (REPLY_TEXT_KEY, 'text')


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


def _check_required_fields(body: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field_name in _STRING_FIELDS:
        value = body.get(field_name)
        if value is None or not isinstance(value, str):
            issues.append(
                _issue(
                    'pack_field_missing',
                    'error',
                    f'{field_name} is required',
                    field=field_name,
                )
            )
            continue
        if not value.strip():
            issues.append(
                _issue(
                    'pack_field_empty', 'error', f'{field_name} must not be empty', field=field_name
                )
            )
        elif len(value) > MAX_FIELD_LEN:
            issues.append(
                _issue(
                    'pack_field_too_long',
                    'error',
                    f'{field_name} exceeds {MAX_FIELD_LEN} characters',
                    field=field_name,
                    detail={'length': len(value), 'max': MAX_FIELD_LEN},
                )
            )
    for field_name in _MAP_FIELDS:
        value = body.get(field_name)
        if value is None:
            continue
        if not isinstance(value, dict):
            issues.append(
                _issue(
                    'pack_field_missing',
                    'error',
                    f'{field_name} must be an object',
                    field=field_name,
                )
            )
            continue
        if len(value) > MAX_MAP_ENTRIES:
            issues.append(
                _issue(
                    'pack_field_too_long',
                    'error',
                    f'{field_name} has more than {MAX_MAP_ENTRIES} entries',
                    field=field_name,
                    detail={'entries': len(value), 'max': MAX_MAP_ENTRIES},
                )
            )
    return issues


def _check_name(name: str | None, *, existing_names: frozenset[str]) -> list[ValidationIssue]:
    if name is None:
        return []
    issues: list[ValidationIssue] = []
    if not NAME_RE.match(name):
        issues.append(_issue('pack_name_invalid', 'error', f'{name!r} is not a valid pack name'))
        return issues
    if name in RESERVED_NAMES:
        issues.append(_issue('pack_name_reserved', 'error', f'{name!r} is a reserved name'))
    if name in existing_names:
        issues.append(_issue('name_conflict', 'error', f'{name!r} is already taken'))
    return issues


def _format_fields(value: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(value) if name}


def _check_placeholders(body: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field_name, required in FORMATTED_PLACEHOLDERS.items():
        value = body.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        try:
            present = _format_fields(value)
        except ValueError:
            issues.append(
                _issue(
                    'pack_template_format_error',
                    'error',
                    f'{field_name} has unbalanced braces',
                    field=field_name,
                )
            )
            continue
        try:
            value.format(**dict.fromkeys(present, ''))
        except (KeyError, IndexError, ValueError):
            issues.append(
                _issue(
                    'pack_template_format_error',
                    'error',
                    f'{field_name} could not be formatted',
                    field=field_name,
                )
            )
        issues.extend(
            _issue(
                'pack_placeholder_missing',
                'error',
                f'{field_name} must contain {{{missing}}}',
                field=field_name,
                detail={'placeholder': missing},
            )
            for missing in sorted(set(required) - present)
        )
        issues.extend(
            _issue(
                'pack_placeholder_unknown',
                'error',
                f'{field_name} uses unknown placeholder {{{extra}}}',
                field=field_name,
                detail={'placeholder': extra},
            )
            for extra in sorted(present - set(required))
        )
    # Verbatim-field warning: a placeholder name literally appearing in a
    # field that is never .format()-ed.
    known_placeholders = {p for req in FORMATTED_PLACEHOLDERS.values() for p in req}
    for field_name in _STRING_FIELDS:
        if field_name in FORMATTED_PLACEHOLDERS:
            continue
        value = body.get(field_name)
        if not isinstance(value, str):
            continue
        issues.extend(
            _issue(
                'pack_placeholder_in_verbatim_field',
                'warning',
                f'{field_name} is sent verbatim but contains {{{placeholder}}}',
                field=field_name,
                detail={'placeholder': placeholder},
            )
            for placeholder in sorted(known_placeholders)
            if f'{{{placeholder}}}' in value
        )
    return issues


def _key_present(text: str, key: str) -> bool:
    return (
        re.search(rf'"{re.escape(key)}"', text) is not None
        or re.search(rf'\b{re.escape(key)}\b', text) is not None
    )


def _check_reply_keys(body: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for contract in REPLY_KEY_CONTRACT.values():
        texts = [str(body.get(f) or '') for f in contract['fields']]
        combined = '\n'.join(texts)
        issues.extend(
            _issue(
                'pack_reply_key_missing',
                'error',
                f'{contract["fields"][-1]} never names the required reply key {key!r}',
                field=contract['fields'][-1],
                detail={'key': key},
            )
            for key in contract['required']
            if not _key_present(combined, key)
        )
    return issues


def check_multi_region_keys(
    body: dict[str, Any], *, max_regions_per_item: int, for_activation: bool
) -> ValidationIssue | None:
    """W8/D-B: the combined calls must name every ``multi_region`` key
    when the profile context calls for more than one box. Warning
    everywhere except activation pairing, where it is an error that
    ``force`` never bypasses."""
    if max_regions_per_item <= 1:
        return None
    missing: list[str] = []
    for call_id in ('combined', 'combined_batch'):
        contract = REPLY_KEY_CONTRACT[call_id]
        texts = [str(body.get(f) or '') for f in contract['fields']]
        combined = '\n'.join(texts)
        for key in contract['multi_region']:
            if not _key_present(combined, key) and key not in missing:
                missing.append(key)
    if not missing:
        return None
    severity = 'error' if for_activation else 'warning'
    return _issue(
        'pack_multi_region_keys_missing',
        severity,
        'The pack does not name every multi-box reply key needed for '
        f'max_regions_per_item={max_regions_per_item}',
        detail={'missing': missing, 'max_regions_per_item': max_regions_per_item},
    )


def _check_text_mode(
    body: dict[str, Any], profile: DetectionProfile | None
) -> list[ValidationIssue]:
    if profile is None:
        return []
    asks_text = any(
        _key_present(str(body.get(f) or ''), key)
        for f in (
            'combined_user_template',
            'combined_batch_rules',
            'combined_batch_system',
            'region_user',
            'region_batch_user',
        )
        for key in _TEXT_ASKING_KEYS
    )
    issues: list[ValidationIssue] = []
    if asks_text and profile.text_reader == 'none':
        issues.append(
            _issue(
                'pack_asks_text_profile_text_free',
                'warning',
                'The pack asks for region text but the profile is text-free',
            )
        )
    if not asks_text and profile.text_reader in ('vlm', 'vlm_then_ocr', 'both'):
        issues.append(
            _issue(
                'pack_no_text_profile_reads_text',
                'warning',
                'The profile reads text via the VLM but the pack never asks for it',
            )
        )
    return issues


def _check_vocabulary(
    body: dict[str, Any], *, class_names: frozenset[str] | None
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if class_names is not None:
        synonyms = body.get('synonyms') or {}
        if isinstance(synonyms, dict):
            issues.extend(
                _issue(
                    'pack_synonym_target_unknown',
                    'warning',
                    f'synonym target {target!r} is not a registry class',
                    field='synonyms',
                    detail={'target': target},
                )
                for target in sorted(set(synonyms.values()))
                if target not in class_names
            )
        descriptions = body.get('class_descriptions') or {}
        if isinstance(descriptions, dict):
            issues.extend(
                _issue(
                    'pack_description_class_unknown',
                    'warning',
                    f'class_descriptions key {key!r} is not a registry class',
                    field='class_descriptions',
                    detail={'class_name': key},
                )
                for key in sorted(descriptions)
                if key not in class_names
            )
    try:
        pack = PromptPack.from_dict({**body, 'name': body.get('name') or 'draft'})
        from src.services.labeling.vlm_prompts import prompt_text_examples

        examples = sorted(prompt_text_examples(pack))
    except Exception:  # pragma: no cover - defensive; malformed draft
        examples = []
    if examples:
        issues.append(
            _issue(
                'pack_example_values',
                'info',
                'These quoted example values may be echoed back as placeholder rejects',
                detail={'examples': examples},
            )
        )
    return issues


def validate_pack(
    name: str | None,
    body: dict[str, Any],
    *,
    existing_names: frozenset[str] = frozenset(),
    profile: DetectionProfile | None = None,
    for_activation: bool = False,
    class_names: frozenset[str] | None = None,
) -> ValidationReport:
    """Validate a prompt-pack envelope (``{name, body}``).

    ``existing_names`` is every currently-taken pack name (across every
    source) EXCLUDING the pack being edited, if any -- callers pass the
    full set for create/clone and an empty set (or one already excluding
    itself) for PUT, where the name cannot change.
    """
    from src.routers.curation._config_common_models import ValidationReport

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    for issue in [
        *_check_name(name, existing_names=existing_names),
        *_check_required_fields(body),
        *_check_placeholders(body),
        *_check_reply_keys(body),
        *_check_text_mode(body, profile),
        *_check_vocabulary(body, class_names=class_names),
    ]:
        (errors if issue.severity == 'error' else warnings).append(issue)

    if profile is not None:
        multi = check_multi_region_keys(
            body, max_regions_per_item=profile.max_regions_per_item, for_activation=for_activation
        )
        if multi is not None:
            (errors if multi.severity == 'error' else warnings).append(multi)

    force_allowed = bool(errors) and all(e.code in BYPASSABLE_CODES for e in errors)
    return ValidationReport(
        ok=not errors, errors=errors, warnings=warnings, force_allowed=force_allowed
    )


__all__ = [
    'BYPASSABLE_CODES',
    'MAX_FIELD_LEN',
    'MAX_MAP_ENTRIES',
    'NAME_RE',
    'RESERVED_NAMES',
    'check_multi_region_keys',
    'validate_pack',
]
