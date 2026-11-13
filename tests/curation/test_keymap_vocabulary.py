"""The served keymap vocabulary carries the Workspace and image editor contexts and actions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.routers.curation._class_hotkeys import validated_hotkey
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
    'workspace.select_all': ('a', 'ctrl+a', 'meta+a'),
}


def test_workspace_actions_are_served() -> None:
    registry = load_registry()
    for action_id, keys in WORKSPACE_KEYS.items():
        assert registry.actions[action_id].default == keys, action_id


def test_the_grid_context_includes_the_workspace_context() -> None:
    registry = load_registry()
    assert registry.active_set('cluster') == {'cluster', 'workspace.items', 'workspace', 'global'}
    assert registry.contexts['workspace'].class_hotkeys_live is True


@pytest.mark.parametrize(
    ('action_id', 'keys'),
    [
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
    assert registry.active_set('table') == {'table', 'workspace.items', 'workspace', 'global'}
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
        assert actions[action_id].context == 'workspace.items', action_id
    assert actions['workspace.select_all'].context == 'workspace'
    assert 'cluster.select_all' not in actions
    assert 'clusters_search.select_all' not in actions
    assert 'table.select_toggle' not in actions
    assert 'table.open_review' not in actions


EDITOR_KEYS = {
    'editor.tool_select': ('v',),
    'editor.tool_draw': ('b',),
    'editor.tool_draw_sticky': ('shift+b',),
    'editor.tool_pan': ('h',),
    'editor.zoom_in': ('=',),
    'editor.zoom_out': ('-',),
    'editor.zoom_fit': ('0',),
    'editor.zoom_actual': ('1',),
    'editor.delete': ('delete', 'backspace'),
    'editor.next_box': ('tab',),
    'editor.prev_box': ('shift+tab',),
    'editor.nudge_up': ('arrowup',),
    'editor.nudge_down_far': ('shift+arrowdown',),
    'editor.resize_left': ('alt+shift+arrowleft',),
    'editor.regions': ('shift+r',),
    'editor.labels': ('shift+l',),
    'editor.undo': ('ctrl+z', 'meta+z'),
    'editor.redo': ('ctrl+shift+z', 'meta+shift+z'),
    'editor.class_picker': ('/',),
    'editor.prev_image': (',',),
    'editor.next_image': ('.',),
    'editor.cancel': ('escape',),
}


def test_editor_actions_are_served_with_their_default_keys() -> None:
    registry = load_registry()
    for action_id, keys in EDITOR_KEYS.items():
        assert registry.actions[action_id].default == keys, action_id


def test_the_editor_context_includes_global_and_keeps_class_hotkeys_live() -> None:
    registry = load_registry()
    assert registry.active_set('editor') == {'editor', 'global'}
    assert registry.contexts['editor'].class_hotkeys_live is True


def test_escape_cancels_and_cannot_be_rebound() -> None:
    assert load_registry().actions['editor.cancel'].modifiable is False


def test_every_other_editor_action_is_modifiable() -> None:
    registry = load_registry()
    locked = [
        a.id
        for a in registry.actions.values()
        if a.context == 'editor' and a.id != 'editor.cancel' and not a.modifiable
    ]
    assert locked == []


def test_the_editor_defaults_collide_with_nothing_in_its_active_set() -> None:
    registry = load_registry()
    active = registry.active_set('editor')
    seen: dict[str, str] = {}
    for action in registry.actions.values():
        if action.context not in active or action.id == 'editor.cancel':
            continue
        for combo in action.default:
            assert combo not in seen, (combo, seen[combo], action.id)
            seen[combo] = action.id


def test_no_editor_default_is_browser_reserved() -> None:
    registry = load_registry()
    reserved = set(registry.grammar.browser_reserved)
    clashes = [
        (a.id, c) for a in registry.actions.values() if a.context == 'editor' for c in a.default
    ]
    assert [x for x in clashes if x[1] in reserved] == []


@pytest.mark.parametrize('letter', ['v', 'b', 'h'])
def test_editor_letters_refuse_a_class_hotkey(letter: str) -> None:
    with pytest.raises(HTTPException) as err:
        validated_hotkey(letter, class_id=None, registry_obj=SimpleNamespace(classes=[]))
    assert err.value.status_code == 422
    detail = err.value.detail
    assert isinstance(detail, dict)
    assert detail['error'] == 'hotkey_reserved'
    assert any(a['context'] == 'editor' for a in detail['actions'])


def test_rebinding_next_box_to_tab_does_not_warn_in_the_editor() -> None:
    report, _, _, _ = validate_keymap(
        {'editor.next_box': ['tab']}, project='default', classes=[], previous_overrides={}
    )
    assert report.ok
    assert [w.code for w in report.warnings] == []


def test_an_editor_key_clash_is_refused() -> None:
    report, _, _, _ = validate_keymap(
        {'editor.tool_pan': ['v']}, project='default', classes=[], previous_overrides={}
    )
    assert not report.ok
    assert [e.code for e in report.errors] == ['keymap_context_collision']


def test_the_class_picker_serves_one_key_per_recent_class() -> None:
    registry = load_registry()
    assert registry.active_set('class_picker') == {'class_picker', 'global'}
    for n in range(1, 10):
        action = registry.actions[f'class_picker.recent_{n}']
        assert (action.context, action.default, action.modifiable) == (
            'class_picker',
            (str(n),),
            True,
        )
        assert str(n) in action.label


def test_the_recent_class_keys_collide_with_nothing() -> None:
    report, _, _, _ = validate_keymap({}, project='default', classes=[], previous_overrides={})
    assert report.ok, [e.code for e in report.errors]
