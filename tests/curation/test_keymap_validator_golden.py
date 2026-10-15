"""Full-output golden for ``validate_keymap`` over a matrix of override maps.

The golden file was captured from the implementation before it was decomposed.
It pins the report (errors and warnings, in order), the class conflicts, the
resolved map and the class-conflict issues for every branch of the validator.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from src.services.curation.keymap_validator import shadowed_conflicts, validate_keymap


GOLDEN = Path(__file__).parent / 'golden' / 'keymap_validator_golden.json'


def _cls(letter: str | None, class_id: int = 7, name: str = 'bmw', *, deprecated: bool = False):
    return {
        'class_id': class_id,
        'class_name': name,
        'hotkey_letter': letter,
        'deprecated': deprecated,
    }


CASES: dict[str, dict[str, Any]] = {
    'empty': {'overrides': {}},
    'unknown_action': {'overrides': {'nope.action': ['q']}},
    'locked_action': {'overrides': {'global.close_overlay': ['q']}},
    'invalid_combo': {'overrides': {'review.skip': ['ctrl+ctrl+x']}},
    'duplicate_combo': {'overrides': {'review.skip': ['q', 'q']}},
    'focus_key_outside_box_edit': {'overrides': {'review.skip': ['tab', 'shift+tab']}},
    'focus_key_in_box_edit': {'overrides': {'box_edit.next_box': ['tab']}},
    'locked_key_elsewhere': {'overrides': {'review.skip': ['escape']}},
    'browser_reserved': {'overrides': {'review.skip': ['ctrl+w']}},
    'too_many_combos': {'overrides': {'review.skip': ['q', 'w', 'e', 'p']}},
    'drop_own_locked_key': {'overrides': {'review.queue.confirm': ['k']}},
    'drop_two_locked_keys': {'overrides': {'review.queue.next': [], 'review.queue.prev': []}},
    'extend_keeps_locked': {'overrides': {'review.queue.confirm': ['enter', 'k']}},
    'context_collision': {'overrides': {'review.skip': ['d']}},
    'collision_two_touched': {
        'overrides': {'review.skip': ['k'], 'review.undo': ['k']},
    },
    'overlay_unbound': {'overrides': {'global.shortcuts_overlay': []}},
    'errors_short_circuit_with_collision': {
        'overrides': {'review.skip': ['d'], 'nope.action': ['q']},
    },
    'class_conflict': {
        'overrides': {'cluster.ignore': ['i']},
        'classes': [_cls('i', 33, 'ice_cream_truck')],
    },
    'class_conflict_multi': {
        'overrides': {'cluster.ignore': ['i'], 'cluster.skip': ['j']},
        'classes': [_cls('i', 33, 'ice'), _cls('j', 34, 'jam')],
    },
    'class_conflict_deprecated_ignored': {
        'overrides': {'cluster.ignore': ['i']},
        'classes': [_cls('i', 33, 'ice', deprecated=True)],
    },
    'class_conflict_not_live_context': {
        'overrides': {'box_edit.grow_right': ['i']},
        'classes': [_cls('i', 33, 'ice')],
    },
    'class_conflict_uppercase_letter': {
        'overrides': {'cluster.ignore': ['i']},
        'classes': [_cls('I', 33, 'ice')],
    },
    'class_conflict_modified_combo_ignored': {
        'overrides': {'cluster.ignore': ['ctrl+i']},
        'classes': [_cls('i', 33, 'ice')],
    },
    'grandfathered_default': {
        'overrides': {},
        'classes': [_cls('b', 12, 'bmw')],
        'previous_overrides': {},
    },
    'grandfathered_via_previous': {
        'overrides': {'cluster.ignore': ['i']},
        'classes': [_cls('i', 33, 'ice')],
        'previous_overrides': {'cluster.ignore': ['i']},
    },
    'no_classes_no_previous': {'overrides': {'cluster.ignore': ['i']}, 'classes': []},
}


def _run(case: dict[str, Any]) -> dict[str, Any]:
    report, conflicts, resolved, issues = validate_keymap(
        case['overrides'],
        project='proj',
        classes=case.get('classes', []),
        previous_overrides=case.get('previous_overrides'),
    )
    return {
        'report': report.model_dump(mode='json'),
        'conflicts': [dataclasses.asdict(c) for c in conflicts],
        'resolved': resolved,
        'issues': [i.model_dump(mode='json') for i in issues],
    }


@pytest.mark.parametrize('name', sorted(CASES))
def test_validate_keymap_matches_golden(name: str) -> None:
    golden = json.loads(GOLDEN.read_text())
    assert _run(CASES[name]) == golden[name]


def test_golden_file_has_exactly_the_cases() -> None:
    assert set(json.loads(GOLDEN.read_text())) - {'__shadowed__'} == set(CASES)


def test_golden_exercises_every_issue_code() -> None:
    golden = json.loads(GOLDEN.read_text())
    codes = {
        i['code']
        for k, g in golden.items()
        if k != '__shadowed__'
        for i in (*g['report']['errors'], *g['report']['warnings'], *g['issues'])
    }
    assert {
        'keymap_unknown_action',
        'keymap_action_locked',
        'keymap_combo_invalid',
        'keymap_key_locked',
        'keymap_browser_reserved',
        'keymap_focus_key',
        'keymap_too_many_combos',
        'keymap_context_collision',
        'keymap_overlay_unbound',
        'keymap_class_hotkey_shadowed',
        'keymap_class_hotkey_conflict',
    } <= codes


def test_shadowed_conflicts_lists_effective_class_hotkeys() -> None:
    issues = shadowed_conflicts('proj', [_cls('b', 12, 'bmw')], {})
    golden = json.loads(GOLDEN.read_text())['__shadowed__']
    assert [i.model_dump(mode='json') for i in issues] == golden
