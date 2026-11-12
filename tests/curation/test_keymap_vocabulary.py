"""The served keymap vocabulary carries the Workspace contexts and actions."""

from __future__ import annotations

import pytest

from src.services.curation.keymap import effective_keys, load_registry
from src.services.curation.keymap_validator import validate_keymap


WORKSPACE_KEYS = {
    'workspace.view_prev': ('[',),
    'workspace.view_next': (']',),
    'workspace.scope_prev': (',',),
    'workspace.scope_next': ('.',),
    'workspace.toggle_details': ('\\',),
    'workspace.open_editor': ('shift+o',),
    'workspace.class_picker': ('/',),
    'workspace.escape': ('escape',),
    'workspace.select_toggle': ('space',),
    'workspace.open_review': ('enter',),
}


def test_workspace_actions_are_served() -> None:
    registry = load_registry()
    for action_id, keys in WORKSPACE_KEYS.items():
        assert registry.actions[action_id].default == keys, action_id


def test_the_grid_context_includes_the_workspace_context() -> None:
    registry = load_registry()
    assert registry.active_set('cluster') == {'cluster', 'workspace', 'global'}
    assert registry.contexts['workspace'].class_hotkeys_live is True


@pytest.mark.parametrize(
    ('action_id', 'keys'),
    [
        ('cluster.select_all', ('a', 'ctrl+a', 'meta+a')),
        ('cluster.confirm', ('shift+enter',)),
        ('cluster.undo', ('z', 'ctrl+z', 'meta+z')),
    ],
)
def test_grid_actions_take_the_workspace_keys(action_id: str, keys: tuple[str, ...]) -> None:
    assert load_registry().actions[action_id].default == keys


def test_the_defaults_collide_nowhere() -> None:
    report, _, _, _ = validate_keymap({}, project='default', classes=[], previous_overrides={})
    assert report.ok, [e.code for e in report.errors]


def test_every_default_action_has_keys() -> None:
    registry = load_registry()
    assert all(effective_keys(a, {}) for a in registry.actions.values())


def test_table_context_serves_row_keys() -> None:
    registry = load_registry()
    assert registry.active_set('table') == {'table', 'workspace', 'global'}
    expected = {
        'table.prev_row': ('arrowup',),
        'table.next_row': ('arrowdown',),
        'table.extend_prev': ('shift+arrowup',),
        'table.extend_next': ('shift+arrowdown',),
    }
    for action_id, keys in expected.items():
        assert registry.actions[action_id].default == keys, action_id


def test_select_and_open_review_are_one_workspace_action_for_every_view() -> None:
    actions = load_registry().actions
    for action_id in ('workspace.select_toggle', 'workspace.open_review'):
        assert actions[action_id].context == 'workspace', action_id
    assert 'table.select_toggle' not in actions
    assert 'table.open_review' not in actions
