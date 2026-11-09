"""Exclude / un-exclude update builders for items (the labeler's Ignore + Undo).

Excluding drops an item's class validation (an excluded item is never a training label) and
flags it ``class_excluded``; every cluster reader filters that flag, so the item's placement in
the serving partition is left alone. Exclude records the pre-exclusion validation in
``excluded_prior_class_validated`` and un-exclude restores it.
"""

from __future__ import annotations

from typing import Any

from src.clients.occ_locks import is_human_marker


PRIOR_VALIDATED = 'excluded_prior_class_validated'


def exclusion_update(current: dict[str, Any], *, reason: str, now: str) -> dict[str, Any]:
    """Update doc excluding ``current``.

    Re-excluding an already-excluded item only refreshes the reason and
    timestamp: its recorded pre-exclusion state must survive, since the
    live fields now hold the excluded placeholders.
    """
    update: dict[str, Any] = {
        'class_excluded': True,
        'excluded_at': now,
        'excluded_by': 'human',
        'excluded_reason': reason,
        'updated_at': now,
    }
    if current.get('class_excluded'):
        return update
    update.update(
        {
            PRIOR_VALIDATED: bool(current.get('class_validated')),
            # An excluded item is not a validated class label.
            'class_validated': False,
        }
    )
    return update


def unexclusion_update(current: dict[str, Any], *, now: str) -> dict[str, Any]:
    """Update doc reversing an exclusion; ``{}`` if ``current`` isn't excluded.

    The validation recorded at exclude time comes back; the item never left its cluster.

    Items excluded before ``excluded_prior_class_validated`` was recorded
    lost their validation flag at exclude time; for those a human-sourced
    class (the only validated class a human exclusion could have wiped
    without a record) is taken as validated. This checks specifically for
    a human write (``is_human_marker``), NOT ``is_locked_class`` — lock
    is a broader "never touch" rule (it also covers ``test_holdout`` and
    validated imports) and is not the same claim as "a human validated
    this", which is what the legacy branch is standing in for. Using the
    broader lock check here would restore a `test_holdout` item with a
    machine-set class as ``class_validated=True`` on un-exclude, which is
    wrong: freezing a holdout item never implies a human validated it.
    """
    if not current.get('class_excluded'):
        return {}
    if PRIOR_VALIDATED in current and current[PRIOR_VALIDATED] is not None:
        validated = bool(current[PRIOR_VALIDATED])
    else:
        validated = is_human_marker(current.get('class_source'))
    class_id = current.get('class_id')
    validated = validated and class_id is not None

    update: dict[str, Any] = {
        'class_excluded': False,
        'excluded_at': None,
        'excluded_by': None,
        'excluded_reason': None,
        PRIOR_VALIDATED: None,
        'class_validated': validated,
        'updated_at': now,
    }
    return update


def park_restored_state_while_excluded(restored: dict[str, Any]) -> dict[str, Any]:
    """Redirect a class-state restore (label Undo) on an excluded item.

    The item stays excluded, so its live validation field keeps its excluded placeholder; the
    restored value becomes the ``excluded_prior_*`` state that un-exclude will apply.
    """
    out = dict(restored)
    out[PRIOR_VALIDATED] = bool(out.pop('class_validated'))
    return out


__all__ = [
    'PRIOR_VALIDATED',
    'exclusion_update',
    'park_restored_state_while_excluded',
    'unexclusion_update',
]
