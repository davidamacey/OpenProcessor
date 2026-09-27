"""W2b keymap validator (CW-K §3.1-§3.4).

Pure function: no OpenSearch I/O beyond the class list the caller
already fetched (per the owner's per-project scope override, §0: class
conflicts are checked against the *bound project's* registry only, not
every project's).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.routers.curation._config_common_models import ValidationIssue, ValidationReport
from src.services.curation.keymap import (
    combo_grammar_error,
    effective_keys,
    is_single_char,
    load_registry,
)


@dataclass(frozen=True)
class ClassConflict:
    project: str
    class_id: int
    class_name: str
    combo: str
    action_id: str


def _class_by_letter(classes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in classes:
        if c.get('deprecated'):
            continue
        letter = (c.get('hotkey_letter') or '').lower()
        if letter:
            out[letter] = c
    return out


def validate_keymap(
    overrides: dict[str, list[str]],
    *,
    project: str,
    classes: list[dict[str, Any]],
    previous_overrides: dict[str, list[str]] | None = None,
) -> tuple[ValidationReport, list[ClassConflict], dict[str, list[str]]]:
    """Validate a proposed override map.

    Returns ``(report, class_conflicts, resolved)`` where ``resolved`` is
    the effective keymap the write would produce. ``report.errors`` only
    ever holds the body-internal (422) codes; a class-hotkey conflict is
    returned separately in ``class_conflicts`` (409), per CW-K §3.1.
    """
    registry = load_registry()
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    previous_overrides = previous_overrides or {}

    # 1-2: unknown action / locked action.
    for action_id, combos in overrides.items():
        field = f'overrides.{action_id}'
        action = registry.actions.get(action_id)
        if action is None:
            errors.append(
                ValidationIssue(
                    code='keymap_unknown_action',
                    severity='error',
                    field=field,
                    message=f"'{action_id}' is not a known action id.",
                    detail={'action_id': action_id},
                )
            )
            continue
        if not action.modifiable:
            errors.append(
                ValidationIssue(
                    code='keymap_action_locked',
                    severity='error',
                    field=field,
                    message=f"'{action_id}' cannot be rebound.",
                    detail={'action_id': action_id},
                )
            )
            continue
        # An action whose default already includes a locked key keeps
        # that key forever (owner rule): it may still be rebound/extended
        # otherwise, but dropping its own locked key or another action
        # claiming that key is always an error.
        own_locked = registry.locked_action_default(action)
        if own_locked and not own_locked.issubset(set(combos)):
            errors.append(
                ValidationIssue(
                    code='keymap_key_locked',
                    severity='error',
                    field=field,
                    message=(f"'{action_id}' cannot lose its locked key(s) {sorted(own_locked)}."),
                    detail={'combo': sorted(own_locked)},
                )
            )

        # 3: grammar.
        if len(combos) > registry.grammar.max_combos_per_action:
            errors.append(
                ValidationIssue(
                    code='keymap_too_many_combos',
                    severity='error',
                    field=field,
                    message=(
                        f"'{action_id}' has {len(combos)} combos; "
                        f'max is {registry.grammar.max_combos_per_action}.'
                    ),
                    detail={
                        'limit': registry.grammar.max_combos_per_action,
                        'requested': len(combos),
                    },
                )
            )
        for combo in combos:
            reason = combo_grammar_error(combo)
            if reason is not None:
                errors.append(
                    ValidationIssue(
                        code='keymap_combo_invalid',
                        severity='error',
                        field=field,
                        message=f"'{combo}' is not a valid combo: {reason}",
                        detail={'combo': combo, 'reason': reason},
                    )
                )
                continue
            if combo in registry.grammar.locked_keys and combo not in own_locked:
                errors.append(
                    ValidationIssue(
                        code='keymap_key_locked',
                        severity='error',
                        field=field,
                        message=f"'{combo}' is a locked key and cannot be bound elsewhere.",
                        detail={'combo': combo},
                    )
                )
            if combo in registry.grammar.browser_reserved:
                errors.append(
                    ValidationIssue(
                        code='keymap_browser_reserved',
                        severity='error',
                        field=field,
                        message=f"'{combo}' is reserved by the browser.",
                        detail={'combo': combo},
                    )
                )
            if combo in {'tab', 'shift+tab'} and action.context != 'box_edit':
                warnings.append(
                    ValidationIssue(
                        code='keymap_focus_key',
                        severity='warning',
                        field=field,
                        message=f"'{combo}' shadows focus navigation outside box editing.",
                        detail={'combo': combo},
                    )
                )

    if errors:
        # Body-internal errors already found -- still compute resolved
        # for the caller's convenience, but skip collision/overlay
        # checks against a map we know is malformed.
        resolved = {aid: effective_keys(a, overrides) for aid, a in registry.actions.items()}
        return (
            ValidationReport(ok=False, errors=errors, warnings=warnings, force_allowed=False),
            [],
            resolved,
        )

    resolved = {aid: effective_keys(a, overrides) for aid, a in registry.actions.items()}

    # 4: context collisions (within each context's active set).
    for context_id in registry.contexts:
        active = registry.active_set(context_id)
        by_combo: dict[str, list[str]] = {}
        for action in registry.actions.values():
            if action.context not in active:
                continue
            for combo in resolved[action.id]:
                by_combo.setdefault(combo, []).append(action.id)
        for combo, action_ids in by_combo.items():
            distinct = sorted(set(action_ids))
            if len(distinct) > 1:
                # Only actually rebound actions produce a *new* error --
                # report once per (context, combo) pair, attributed to
                # every action in the request that touches this combo.
                touched = [a for a in distinct if a in overrides]
                errors.extend(
                    ValidationIssue(
                        code='keymap_context_collision',
                        severity='error',
                        field=f'overrides.{action_id}',
                        message=f"'{combo}' is already bound in context '{context_id}'.",
                        detail={'combo': combo, 'context': context_id, 'action_ids': distinct},
                    )
                    for action_id in touched
                )

    # 5: the overlay must always have a key.
    overlay = resolved.get('global.shortcuts_overlay', [])
    if not overlay:
        errors.append(
            ValidationIssue(
                code='keymap_overlay_unbound',
                severity='error',
                field='overrides.global.shortcuts_overlay',
                message='The shortcuts overlay must always have at least one key.',
                detail={},
            )
        )

    # 6: no-confirm warning.
    warnings.extend(
        ValidationIssue(
            code='keymap_context_no_confirm',
            severity='warning',
            field=f'overrides.{action.id}',
            message=f"Context '{action.context}' has no key for its confirm action.",
            detail={'context': action.context},
        )
        for action in registry.actions.values()
        if action.group == 'confirm' and not resolved[action.id]
    )

    if errors:
        return (
            ValidationReport(ok=False, errors=errors, warnings=warnings, force_allowed=False),
            [],
            resolved,
        )

    # 7: class-hotkey conflicts (409, separate from the 422 report) --
    # only single-char unmodified combos in a class_hotkeys_live context
    # can collide with a class hotkey (§3.4/§2.3).
    by_letter = _class_by_letter(classes)
    class_conflicts: list[ClassConflict] = []
    for action in registry.actions.values():
        if not registry.contexts[action.context].class_hotkeys_live:
            continue
        for combo in resolved[action.id]:
            letter = is_single_char(combo)
            if letter is None or letter not in by_letter:
                continue
            cls = by_letter[letter]
            # Grandfathered: the combo was already effective for this
            # action before the write -- report as a warning, not a
            # blocking conflict.
            if letter in effective_keys(action, previous_overrides):
                warnings.append(
                    ValidationIssue(
                        code='keymap_class_hotkey_shadowed',
                        severity='warning',
                        field=f'overrides.{action.id}',
                        message=(
                            f"Class '{cls['class_name']}' (hotkey '{letter}') shares "
                            f"'{letter}' with {action.label} on {action.context}."
                        ),
                        detail={
                            'combo': letter,
                            'action_id': action.id,
                            'conflicts': [
                                {
                                    'project': project,
                                    'class_id': cls['class_id'],
                                    'class_name': cls['class_name'],
                                }
                            ],
                        },
                    )
                )
                continue
            class_conflicts.append(
                ClassConflict(
                    project=project,
                    class_id=cls['class_id'],
                    class_name=cls['class_name'],
                    combo=letter,
                    action_id=action.id,
                )
            )

    report = ValidationReport(ok=not errors, errors=errors, warnings=warnings, force_allowed=False)
    return report, class_conflicts, resolved


def shadowed_conflicts(
    project: str, classes: list[dict[str, Any]], overrides: dict[str, list[str]]
) -> list[ValidationIssue]:
    """Pre-existing (grandfathered) shadow warnings for a plain ``GET`` --
    every currently-effective class hotkey that collides with the
    *current* stored ``overrides`` (no proposed write)."""
    registry = load_registry()
    by_letter = _class_by_letter(classes)
    issues: list[ValidationIssue] = []
    for action in registry.actions.values():
        if not registry.contexts[action.context].class_hotkeys_live:
            continue
        for combo in effective_keys(action, overrides):
            letter = is_single_char(combo)
            if letter is None or letter not in by_letter:
                continue
            cls = by_letter[letter]
            issues.append(
                ValidationIssue(
                    code='keymap_class_hotkey_shadowed',
                    severity='warning',
                    field=f'overrides.{action.id}',
                    message=(
                        f"Class '{cls['class_name']}' (hotkey '{letter}') shares "
                        f"'{letter}' with {action.label} on {action.context}."
                    ),
                    detail={
                        'combo': letter,
                        'action_id': action.id,
                        'conflicts': [
                            {
                                'project': project,
                                'class_id': cls['class_id'],
                                'class_name': cls['class_name'],
                            }
                        ],
                    },
                )
            )
    return issues


__all__ = ['ClassConflict', 'shadowed_conflicts', 'validate_keymap']
