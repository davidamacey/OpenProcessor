"""Validation of a prompt pack's ``attributes`` and the assist/attributes calls.

Run by :func:`~src.services.config_store.pack_validation.validate_pack`. The
attribute list is checked entry by entry (key shape, uniqueness, limits), and
the prompt fields the declared attributes need must be filled: ``assist_*`` and
``attributes_*`` for item attributes, ``attributes_*`` for box attributes.
"""

from __future__ import annotations

from typing import Any

from src.schemas.curation.config_common import ValidationIssue
from src.services.labeling.vlm_attributes import (
    APPLIES_TO,
    KEY_RE,
    MAX_ATTRIBUTES,
    MAX_DESCRIPTION_LEN,
    MAX_VALUE_LEN,
    MAX_VALUES,
)


_ITEM_FIELDS = ('assist_system', 'assist_user_template')
_ATTRIBUTE_FIELDS = ('attributes_system', 'attributes_user_template')


def _error(code: str, message: str, field: str, **detail: Any) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity='error',
        field=field,
        message=message,
        detail=detail,
    )


def _check_entry(index: int, entry: Any, seen: set[str]) -> list[ValidationIssue]:
    where = f'attributes[{index}]'
    if not isinstance(entry, dict):
        return [_error('pack_attribute_invalid', f'{where} must be an object', where)]
    issues: list[ValidationIssue] = []
    key = entry.get('key')
    if not isinstance(key, str) or not KEY_RE.match(key):
        issues.append(
            _error(
                'pack_attribute_invalid',
                f'{where}.key must match {KEY_RE.pattern}',
                f'{where}.key',
            )
        )
    elif key in seen:
        issues.append(
            _error('pack_attribute_duplicate', f'attribute key {key!r} is declared twice', where)
        )
    else:
        seen.add(key)
    description = entry.get('description', '')
    if not isinstance(description, str) or len(description) > MAX_DESCRIPTION_LEN:
        issues.append(
            _error(
                'pack_attribute_invalid',
                f'{where}.description must be a string of at most {MAX_DESCRIPTION_LEN} characters',
                f'{where}.description',
            )
        )
    if entry.get('applies_to', 'item') not in APPLIES_TO:
        issues.append(
            _error(
                'pack_attribute_invalid',
                f'{where}.applies_to must be one of {", ".join(APPLIES_TO)}',
                f'{where}.applies_to',
            )
        )
    values = entry.get('values', [])
    bad_values = (
        not isinstance(values, list)
        or len(values) > MAX_VALUES
        or any(not isinstance(v, str) or not v.strip() or len(v) > MAX_VALUE_LEN for v in values)
    )
    if bad_values:
        issues.append(
            _error(
                'pack_attribute_invalid',
                f'{where}.values must be at most {MAX_VALUES} non-empty strings of at most '
                f'{MAX_VALUE_LEN} characters',
                f'{where}.values',
            )
        )
    return issues


def _missing_prompts(
    body: dict[str, Any], names: tuple[str, ...], why: str
) -> list[ValidationIssue]:
    return [
        _error(
            'pack_field_empty',
            f'{name} must not be empty: the pack declares {why}',
            name,
        )
        for name in names
        if not str(body.get(name) or '').strip()
    ]


def check_attributes(body: dict[str, Any]) -> list[ValidationIssue]:
    raw = body.get('attributes')
    if raw is None:
        return []
    if not isinstance(raw, list):
        return [_error('pack_attribute_invalid', 'attributes must be a list', 'attributes')]
    if len(raw) > MAX_ATTRIBUTES:
        return [
            _error(
                'pack_field_too_long',
                f'attributes has more than {MAX_ATTRIBUTES} entries',
                'attributes',
                entries=len(raw),
                max=MAX_ATTRIBUTES,
            )
        ]
    seen: set[str] = set()
    issues: list[ValidationIssue] = []
    for index, entry in enumerate(raw):
        issues.extend(_check_entry(index, entry, seen))
    applies = {e.get('applies_to', 'item') for e in raw if isinstance(e, dict)}
    if 'item' in applies:
        issues.extend(_missing_prompts(body, _ITEM_FIELDS, 'item attributes'))
        issues.extend(_missing_prompts(body, _ATTRIBUTE_FIELDS, 'item attributes'))
    elif 'box' in applies:
        issues.extend(_missing_prompts(body, _ATTRIBUTE_FIELDS, 'box attributes'))
    return issues


__all__ = ['check_attributes']
