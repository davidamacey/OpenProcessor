"""Per-class dataset thresholds — one definition, enforced by the training
preflight and served to clients (``/train/preflight``, ``/stats/classes``,
``/classes``, ``/test_holdout/stats``) so no client keeps its own copy.
Leaf module.
"""

from __future__ import annotations

from typing import Literal


HARD_MIN_CROPS_PER_CLASS = 20
"""Preflight blocks training a class with fewer validated crops."""

WARN_MIN_CROPS_PER_CLASS = 500
"""Preflight warns below this many validated crops."""

MIN_TEST_CROPS_PER_CLASS = 5
"""Every class in the frozen test holdout gets at least this many crops
(or all of them, if it has fewer); preflight warns below it."""

AUG_TARGET_MIN = 500
AUG_TARGET_MAX = 3000
"""A class's augmentation target is its validated count clamped to
``[AUG_TARGET_MIN, AUG_TARGET_MAX]``: thin classes are augmented up to the
floor, big ones are capped."""

Adequacy = Literal['ok', 'warn', 'block']


def adequacy(validated: int) -> Adequacy:
    if validated < HARD_MIN_CROPS_PER_CLASS:
        return 'block'
    if validated < WARN_MIN_CROPS_PER_CLASS:
        return 'warn'
    return 'ok'


def aug_target(validated: int) -> int:
    return min(max(validated, AUG_TARGET_MIN), AUG_TARGET_MAX)


def dataset_thresholds() -> dict[str, int]:
    return {
        'block_below': HARD_MIN_CROPS_PER_CLASS,
        'warn_below': WARN_MIN_CROPS_PER_CLASS,
        'min_test_per_class': MIN_TEST_CROPS_PER_CLASS,
        'aug_target_min': AUG_TARGET_MIN,
        'aug_target_max': AUG_TARGET_MAX,
    }


__all__ = [
    'AUG_TARGET_MAX',
    'AUG_TARGET_MIN',
    'HARD_MIN_CROPS_PER_CLASS',
    'MIN_TEST_CROPS_PER_CLASS',
    'WARN_MIN_CROPS_PER_CLASS',
    'Adequacy',
    'adequacy',
    'aug_target',
    'dataset_thresholds',
]
