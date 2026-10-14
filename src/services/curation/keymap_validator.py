"""Keymap validator.

Pure function: no OpenSearch I/O beyond the class list the caller
already fetched (per the owner's per-project scope override, §0: class
conflicts are checked against the *bound project's* registry only, not
every project's).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.schemas.curation.config_common import ValidationIssue, ValidationReport
from src.services.curation.keymap import (
    ActionRegistry,
    KeymapAction,
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


def _error(code: str, field: str, message: str, detail: dict[str, Any]) -> ValidationIssue:
    return ValidationIssue(code=code, severity='error', field=field, message=message, detail=detail)


def _warning(code: str, field: str, message: str, detail: dict[str, Any]) -> ValidationIssue:
    return ValidationIssue(
        code=code, severity='warning', field=field, message=message, detail=detail
    )


def _hotkey_conflict_detail(
    project: str, cls: dict[str, Any], letter: str, action_id: str
) -> dict[str, Any]:
    return {
        'combo': letter,
        'action_id': action_id,
        'conflicts': [
            {
                'project': project,
                'class_id': cls['class_id'],
                'class_name': cls['class_name'],
            }
        ],
    }


def _combo_issues(
    registry: ActionRegistry,
    action: KeymapAction,
    combo: str,
    field: str,
    own_locked: set[str],
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    """Errors and warnings for one combo of a rebindable action."""
    reason = combo_grammar_error(combo)
    if reason is not None:
        return [
            _error(
                'keymap_combo_invalid',
                field,
                f"'{combo}' is not a valid combo: {reason}",
                {'combo': combo, 'reason': reason},
            )
        ], []
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    if combo in registry.grammar.locked_keys and combo not in own_locked:
        errors.append(
            _error(
                'keymap_key_locked',
                field,
                f"'{combo}' is a locked key and cannot be bound elsewhere.",
                {'combo': combo},
            )
        )
    if combo in registry.grammar.browser_reserved:
        errors.append(
            _error(
                'keymap_browser_reserved',
                field,
                f"'{combo}' is reserved by the browser.",
                {'combo': combo},
            )
        )
    if combo in {'tab', 'shift+tab'} and action.context != 'box_edit':
        warnings.append(
            _warning(
                'keymap_focus_key',
                field,
                f"'{combo}' shadows focus navigation outside box editing.",
                {'combo': combo},
            )
        )
    return errors, warnings


def _action_issues(
    registry: ActionRegistry, action_id: str, combos: list[str]
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    """Body-internal errors and warnings for one override entry (steps 1-3)."""
    field = f'overrides.{action_id}'
    action = registry.actions.get(action_id)
    if action is None:
        return [
            _error(
                'keymap_unknown_action',
                field,
                f"'{action_id}' is not a known action id.",
                {'action_id': action_id},
            )
        ], []
    if not action.modifiable:
        return [
            _error(
                'keymap_action_locked',
                field,
                f"'{action_id}' cannot be rebound.",
                {'action_id': action_id},
            )
        ], []
    own_locked = registry.locked_action_default(action)
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    # Grammar is checked before too_many_combos, per §3.1's order.
    if len(set(combos)) != len(combos):
        errors.append(
            _error(
                'keymap_combo_invalid',
                field,
                f"'{action_id}' repeats a combo.",
                {'combo': combos, 'reason': 'duplicate combo'},
            )
        )
    for combo in combos:
        combo_errors, combo_warnings = _combo_issues(registry, action, combo, field, own_locked)
        errors.extend(combo_errors)
        warnings.extend(combo_warnings)

    if len(combos) > registry.grammar.max_combos_per_action:
        errors.append(
            _error(
                'keymap_too_many_combos',
                field,
                (
                    f"'{action_id}' has {len(combos)} combos; "
                    f'max is {registry.grammar.max_combos_per_action}.'
                ),
                {
                    'limit': registry.grammar.max_combos_per_action,
                    'requested': len(combos),
                },
            )
        )

    # An action whose default already includes a locked key keeps
    # that key forever (owner rule): it may still be rebound/extended
    # otherwise, but dropping its own locked key is always an error.
    # One issue per lost key (minor: keep `detail.combo` a string in
    # every keymap_key_locked issue, not a list in this one case).
    errors.extend(
        _error(
            'keymap_key_locked',
            field,
            f"'{action_id}' cannot lose its locked key '{lost}'.",
            {'combo': lost},
        )
        for lost in sorted(own_locked - set(combos))
    )
    return errors, warnings


def _collision_errors(
    registry: ActionRegistry,
    resolved: dict[str, list[str]],
    overrides: dict[str, list[str]],
) -> list[ValidationIssue]:
    """Step 4: context collisions (within each context's active set)."""
    errors: list[ValidationIssue] = []
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
                    _error(
                        'keymap_context_collision',
                        f'overrides.{action_id}',
                        f"'{combo}' is already bound in context '{context_id}'.",
                        {'combo': combo, 'context': context_id, 'action_ids': distinct},
                    )
                    for action_id in touched
                )
    return errors


def _resolved_keys(
    registry: ActionRegistry, overrides: dict[str, list[str]]
) -> dict[str, list[str]]:
    return {aid: effective_keys(a, overrides) for aid, a in registry.actions.items()}


def _resolved_issues(
    registry: ActionRegistry,
    resolved: dict[str, list[str]],
    overrides: dict[str, list[str]],
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    """Errors and warnings that need the resolved map (steps 4-6)."""
    errors = _collision_errors(registry, resolved, overrides)

    # 5: the overlay must always have a key.
    if not resolved.get('global.shortcuts_overlay', []):
        errors.append(
            _error(
                'keymap_overlay_unbound',
                'overrides.global.shortcuts_overlay',
                'The shortcuts overlay must always have at least one key.',
                {},
            )
        )

    # 6: no-confirm warning.
    warnings = [
        _warning(
            'keymap_context_no_confirm',
            f'overrides.{action.id}',
            f"Context '{action.context}' has no key for its confirm action.",
            {'context': action.context},
        )
        for action in registry.actions.values()
        if action.group == 'confirm' and not resolved[action.id]
    ]
    return errors, warnings


def _class_hotkey_findings(
    registry: ActionRegistry,
    resolved: dict[str, list[str]],
    *,
    project: str,
    classes: list[dict[str, Any]],
    previous_overrides: dict[str, list[str]],
) -> tuple[list[ClassConflict], list[ValidationIssue], list[ValidationIssue]]:
    """Step 7: class-hotkey conflicts (409, separate from the 422 report) --
    only single-char unmodified combos in a class_hotkeys_live context
    can collide with a class hotkey (§3.4/§2.3).

    Returns ``(conflicts, conflict_issues, shadow_warnings)``.
    """
    by_letter = _class_by_letter(classes)
    class_conflicts: list[ClassConflict] = []
    class_conflict_issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
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
                warnings.append(_shadow_warning(project, cls, letter, action))
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
            # M3: a class-hotkey conflict is a real issue, not just a
            # separate structured field -- POST /keymap/validate must
            # answer ok:false with it listed, and the 409's report must
            # carry it too. Kept OUT of `errors` (which stays
            # body-internal-only) so PUT/reset can still tell "422
            # validation_failed" apart from "409 class_hotkey_conflict"
            # -- see class_conflict_issues below.
            class_conflict_issues.append(
                _error(
                    'keymap_class_hotkey_conflict',
                    f'overrides.{action.id}',
                    f"'{letter}' is bound to class '{cls['class_name']}' in project {project}.",
                    _hotkey_conflict_detail(project, cls, letter, action.id),
                )
            )
    return class_conflicts, class_conflict_issues, warnings


def _shadow_warning(
    project: str, cls: dict[str, Any], letter: str, action: KeymapAction
) -> ValidationIssue:
    return _warning(
        'keymap_class_hotkey_shadowed',
        f'overrides.{action.id}',
        (
            f"Class '{cls['class_name']}' (hotkey '{letter}') shares "
            f"'{letter}' with {action.label} on {action.context}."
        ),
        _hotkey_conflict_detail(project, cls, letter, action.id),
    )


def validate_keymap(
    overrides: dict[str, list[str]],
    *,
    project: str,
    classes: list[dict[str, Any]],
    previous_overrides: dict[str, list[str]] | None = None,
) -> tuple[ValidationReport, list[ClassConflict], dict[str, list[str]], list[ValidationIssue]]:
    """Validate a proposed override map.

    Returns ``(report, class_conflicts, resolved, class_conflict_issues)``.
    ``report.errors`` only ever holds the body-internal (422) codes, so a
    PUT/reset caller can still tell "422 validation_failed" apart from
    "409 class_hotkey_conflict". ``class_conflict_issues`` carries the
    same conflicts as ``keymap_class_hotkey_conflict``
    :class:`ValidationIssue` objects (CW-K §3.1/M3) -- ``POST
    /keymap/validate`` folds them into its own ``ok``/``errors`` (it
    never 409s), and the 409 error body's ``report`` carries them too.
    """
    registry = load_registry()
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    previous_overrides = previous_overrides or {}

    # 1-3: unknown action / locked action / grammar.
    for action_id, combos in overrides.items():
        action_errors, action_warnings = _action_issues(registry, action_id, combos)
        errors.extend(action_errors)
        warnings.extend(action_warnings)

    resolved = _resolved_keys(registry, overrides)
    if errors:
        # Body-internal errors already found -- still compute resolved
        # for the caller's convenience, but skip collision/overlay
        # checks against a map we know is malformed.
        return _rejected(errors, warnings, resolved)

    resolved_errors, resolved_warnings = _resolved_issues(registry, resolved, overrides)
    errors.extend(resolved_errors)
    warnings.extend(resolved_warnings)
    if errors:
        return _rejected(errors, warnings, resolved)

    class_conflicts, class_conflict_issues, shadow_warnings = _class_hotkey_findings(
        registry,
        resolved,
        project=project,
        classes=classes,
        previous_overrides=previous_overrides,
    )
    warnings.extend(shadow_warnings)
    report = ValidationReport(ok=True, errors=errors, warnings=warnings, force_allowed=False)
    return report, class_conflicts, resolved, class_conflict_issues


def _rejected(
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
    resolved: dict[str, list[str]],
) -> tuple[ValidationReport, list[ClassConflict], dict[str, list[str]], list[ValidationIssue]]:
    report = ValidationReport(ok=False, errors=errors, warnings=warnings, force_allowed=False)
    return report, [], resolved, []


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
            issues.append(_shadow_warning(project, by_letter[letter], letter, action))
    return issues


__all__ = ['ClassConflict', 'shadowed_conflicts', 'validate_keymap']
