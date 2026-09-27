"""W2b: the per-project configurable keymap -- action registry, grammar
validator, reserved-hotkey derivation, and the ``GET/PUT/validate/reset
{prefix}/keymap`` routes (CW-K §0/§2/§3/§4)."""

from __future__ import annotations

from typing import Any

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.services.curation.keymap import (
    combo_grammar_error,
    effective_keys,
    load_registry,
    reserved_hotkeys,
)
from src.services.curation.keymap_validator import validate_keymap


LEGACY_RESERVED_HOTKEY_LETTERS = frozenset('gndzxuam/feb')


# --------------------------------------------------------------- registry


def test_registry_loads_and_w8_rows_present() -> None:
    registry = load_registry()
    for action_id in ('review.region.accept_box', 'review.region.reject_box', 'box_edit.next_box'):
        assert action_id in registry.actions
    assert registry.actions['review.region.accept_box'].default == ('y',)
    assert registry.actions['review.region.reject_box'].default == ('r',)


def test_active_set_transitive_closure() -> None:
    registry = load_registry()
    active = registry.active_set('review.queue')
    assert active == {'review.queue', 'review', 'global'}
    assert 'box_edit' not in active


def test_grammar_rejects_unrecognized_combo() -> None:
    assert combo_grammar_error('ctrl+shift+z') is None
    assert combo_grammar_error('') is not None
    assert combo_grammar_error('f13') is not None
    assert combo_grammar_error('gg') is not None


# ------------------------------------------------------- reserved_hotkeys


def test_derived_default_equals_legacy_set_plus_w8_and_overlay() -> None:
    """CW-K §3.4: the derived default = the legacy hardcoded set UNION {y, r, backtick}."""
    assert set(reserved_hotkeys({})) == LEGACY_RESERVED_HOTKEY_LETTERS | {'y', 'r', '`'}


def test_moving_an_override_leaves_shared_letter_reserved_and_adds_new_one() -> None:
    """Moving review.queue.discard off 'd' to 'k' leaves 'd' reserved
    (still bound by review.region.reject / a live context) and adds 'k'."""
    before = set(reserved_hotkeys({}))
    assert 'd' in before
    after = set(reserved_hotkeys({'review.queue.discard': ['k']}))
    assert 'd' in after  # review.region.reject still holds it
    assert 'k' in after
    assert after == before | {'k'}


# ------------------------------------------------------------- validator


def _classes(
    *, letter: str | None = None, class_id: int = 1, name: str = 'sedan'
) -> list[dict[str, Any]]:
    return [
        {'class_id': class_id, 'class_name': name, 'hotkey_letter': letter, 'deprecated': False}
    ]


def test_unknown_action_is_422() -> None:
    report, conflicts, _ = validate_keymap(
        {'nope.nope': ['x']}, project='default', classes=[], previous_overrides={}
    )
    assert not report.ok
    assert report.errors[0].code == 'keymap_unknown_action'
    assert conflicts == []


def test_locked_action_cannot_be_overridden() -> None:
    report, _, _ = validate_keymap(
        {'global.close_overlay': ['x']}, project='default', classes=[], previous_overrides={}
    )
    assert not report.ok
    assert report.errors[0].code == 'keymap_action_locked'


def test_too_many_combos() -> None:
    report, _, _ = validate_keymap(
        {'cluster.ignore': ['k', 'l', 'p', 'o']},
        project='default',
        classes=[],
        previous_overrides={},
    )
    assert not report.ok
    assert report.errors[0].code == 'keymap_too_many_combos'


def test_locked_key_cannot_move_to_another_action() -> None:
    report, _, _ = validate_keymap(
        {'cluster.ignore': ['escape']}, project='default', classes=[], previous_overrides={}
    )
    assert not report.ok
    assert report.errors[0].code == 'keymap_key_locked'


def test_browser_reserved_combo_rejected() -> None:
    report, _, _ = validate_keymap(
        {'review.skip': ['ctrl+w']}, project='default', classes=[], previous_overrides={}
    )
    assert not report.ok
    assert report.errors[0].code == 'keymap_browser_reserved'


def test_context_collision_within_active_set() -> None:
    # review.undo (z) and cluster.discard (x) don't collide (different
    # active sets); moving cluster.ignore onto cluster.discard's 'x' does.
    report, _, _ = validate_keymap(
        {'cluster.ignore': ['x']}, project='default', classes=[], previous_overrides={}
    )
    assert not report.ok
    assert report.errors[0].code == 'keymap_context_collision'
    assert set(report.errors[0].detail['action_ids']) == {'cluster.ignore', 'cluster.discard'}


def test_overlay_cannot_be_left_with_no_key() -> None:
    report, _, _ = validate_keymap(
        {'global.shortcuts_overlay': []}, project='default', classes=[], previous_overrides={}
    )
    assert not report.ok
    assert report.errors[0].code == 'keymap_overlay_unbound'


def test_class_hotkey_conflict_is_reported_separately_from_422_errors() -> None:
    report, conflicts, _ = validate_keymap(
        {'cluster.ignore': ['i']},
        project='default',
        classes=_classes(letter='i', class_id=33, name='ice_cream_truck'),
        previous_overrides={},
    )
    assert report.ok  # not a body-internal error
    assert len(conflicts) == 1
    assert conflicts[0].class_id == 33
    assert conflicts[0].combo == 'i'
    assert conflicts[0].action_id == 'cluster.ignore'


def test_preexisting_class_conflict_is_grandfathered_as_warning() -> None:
    """A class already bound to 'b' before this write (which review.region.back
    already used) produces a warning, not a blocking conflict."""
    previous: dict[str, list[str]] = {}
    classes = _classes(letter='b', class_id=12, name='bmw')
    report, conflicts, _ = validate_keymap(
        {'review.region.back': ['b']},
        project='default',
        classes=classes,
        previous_overrides=previous,
    )
    assert report.ok
    assert conflicts == []
    assert any(w.code == 'keymap_class_hotkey_shadowed' for w in report.warnings)


def test_delta22_agreement_region_keys_subset_of_reserved() -> None:
    """CW-K §3.5: every single-char review.region.* key is in
    reserved_hotkeys, and every reserved_hotkeys entry is bound by some
    action in a class_hotkeys_live context."""
    registry = load_registry()
    reserved = set(reserved_hotkeys({}))
    for action in registry.actions.values():
        if action.context != 'review.region':
            continue
        for combo in action.default:
            if len(combo) == 1:
                assert combo in reserved
    live_letters: set[str] = set()
    for action in registry.actions.values():
        if not registry.contexts[action.context].class_hotkeys_live:
            continue
        for combo in effective_keys(action, {}):
            if len(combo) == 1:
                live_letters.add(combo)
    assert reserved == live_letters


# ----------------------------------------------------------------- routes


@pytest.fixture
def registry(tmp_path: Any) -> ClassRegistry:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('sedan', group='vehicle')
    return reg


@pytest.fixture
def fake_os() -> FakeConfigOpenSearch:
    return FakeConfigOpenSearch()


@pytest.fixture
def client(registry: ClassRegistry, fake_os: FakeConfigOpenSearch, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


PREFIX = '/curation/projects/default'


def test_get_keymap_no_doc_serves_defaults(client: TestClient) -> None:
    r = client.get(f'{PREFIX}/keymap')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['scope'] == 'project'
    assert body['project'] == 'default'
    assert body['is_default'] is True
    assert body['revision'] == 0
    assert body['overrides'] == {}
    assert set(body['reserved_hotkeys']) == LEGACY_RESERVED_HOTKEY_LETTERS | {'y', 'r', '`'}
    action_ids = {a['id'] for a in body['actions']}
    assert 'review.region.accept_box' in action_ids
    assert 'box_edit.next_box' in action_ids


def test_glue_g3_region_actions_unavailable_with_no_region_profile(client: TestClient) -> None:
    body = client.get(f'{PREFIX}/keymap').json()
    by_id = {a['id']: a for a in body['actions']}
    assert by_id['review.region.accept_box']['available'] is False
    assert by_id['box_edit.next_box']['available'] is False
    assert by_id['global.shortcuts_overlay']['available'] is True


def test_put_keymap_occ_success_then_stale_conflict(client: TestClient) -> None:
    r = client.put(
        f'{PREFIX}/keymap',
        json={'expected_revision': 0, 'overrides': {'cluster.ignore': ['k']}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['revision'] == 1
    assert body['overrides'] == {'cluster.ignore': ['k']}
    assert body['is_default'] is False

    stale = client.put(
        f'{PREFIX}/keymap',
        json={'expected_revision': 0, 'overrides': {'cluster.ignore': ['l']}},
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()['detail']['error'] == 'revision_conflict'
    assert stale.json()['detail']['current_revision'] == 1


def test_put_keymap_422_on_collision() -> None:
    pass  # covered at the validator level; route-level 422 below


def test_put_keymap_422_body_internal(client: TestClient) -> None:
    r = client.put(
        f'{PREFIX}/keymap',
        json={'expected_revision': 0, 'overrides': {'cluster.ignore': ['x']}},
    )
    assert r.status_code == 422, r.text
    assert r.json()['detail']['error'] == 'validation_failed'
    assert r.json()['detail']['report']['errors'][0]['code'] == 'keymap_context_collision'


def test_put_keymap_class_conflict_409_then_unbind(
    client: TestClient, registry: ClassRegistry
) -> None:
    class_id = registry.load().classes[0].class_id
    registry.load().classes[0].hotkey_letter = 'i'
    registry._atomic_write(registry.load())

    conflict = client.put(
        f'{PREFIX}/keymap',
        json={'expected_revision': 0, 'overrides': {'cluster.ignore': ['i']}},
    )
    assert conflict.status_code == 409, conflict.text
    detail = conflict.json()['detail']
    assert detail['error'] == 'class_hotkey_conflict'
    assert detail['class_conflicts'][0]['class_id'] == class_id
    # Not written.
    entry = registry.get(class_id)
    assert entry is not None
    assert entry.hotkey_letter == 'i'

    unbound = client.put(
        f'{PREFIX}/keymap',
        json={
            'expected_revision': 0,
            'overrides': {'cluster.ignore': ['i']},
            'unbind_conflicting_class_hotkeys': True,
        },
    )
    assert unbound.status_code == 200, unbound.text
    body = unbound.json()
    assert body['unbound_class_hotkeys'][0]['class_id'] == class_id
    assert body['unbound_class_hotkeys'][0]['was'] == 'i'
    entry = registry.get(class_id)
    assert entry is not None
    assert entry.hotkey_letter is None


def test_validate_route_never_4xx_and_reports(client: TestClient) -> None:
    r = client.post(f'{PREFIX}/keymap/validate', json={'overrides': {'cluster.ignore': ['x']}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['ok'] is False
    assert body['errors'][0]['code'] == 'keymap_context_collision'
    assert 'resolved' in body


def test_reset_specific_action_then_reset_all(client: TestClient) -> None:
    put = client.put(
        f'{PREFIX}/keymap',
        json={'expected_revision': 0, 'overrides': {'cluster.ignore': ['k'], 'review.skip': ['j']}},
    )
    assert put.status_code == 200, put.text
    rev = put.json()['revision']

    reset_one = client.post(
        f'{PREFIX}/keymap/reset',
        json={'expected_revision': rev, 'action_ids': ['cluster.ignore']},
    )
    assert reset_one.status_code == 200, reset_one.text
    assert reset_one.json()['overrides'] == {'review.skip': ['j']}

    rev2 = reset_one.json()['revision']
    reset_all = client.post(f'{PREFIX}/keymap/reset', json={'expected_revision': rev2})
    assert reset_all.status_code == 200, reset_all.text
    assert reset_all.json()['overrides'] == {}
    assert reset_all.json()['is_default'] is True


def test_classes_route_serves_derived_reserved_hotkeys(client: TestClient) -> None:
    body = client.get(f'{PREFIX}/classes').json()
    assert set(body['reserved_hotkeys']) == LEGACY_RESERVED_HOTKEY_LETTERS | {'y', 'r', '`'}


def test_class_put_rejects_reserved_hotkey_as_config_error_detail(client: TestClient) -> None:
    class_id = 0
    r = client.put(f'{PREFIX}/classes/{class_id}', json={'hotkey_letter': 'd'})
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    assert detail['error'] == 'hotkey_reserved'
    assert any(a['action_id'] == 'review.queue.discard' for a in detail['actions'])


def test_class_put_rejects_taken_hotkey_as_config_error_detail(
    client: TestClient, registry: ClassRegistry
) -> None:
    second_id = registry.add_class('suv', group='vehicle')
    reg = registry.load()
    for c in reg.classes:
        if c.class_id == 0:
            c.hotkey_letter = 's'
    registry._atomic_write(reg)

    r = client.put(f'{PREFIX}/classes/{second_id}', json={'hotkey_letter': 's'})
    assert r.status_code == 409, r.text
    detail = r.json()['detail']
    assert detail['error'] == 'hotkey_taken'
    assert detail['class_id'] == 0


def test_config_changed_event_published_once_on_put(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import event_hub

    monkeypatch.setenv('OP_EVENT_BUS', 'process')
    event_hub._HUB = None

    async def _run() -> None:
        hub = event_hub.get_event_hub()
        sub = await hub.subscribe(project='default')
        client.put(
            f'{PREFIX}/keymap',
            json={'expected_revision': 0, 'overrides': {'cluster.ignore': ['k']}},
        )
        event = sub.queue.get_nowait()
        assert event['type'] == 'config.changed'
        assert event['axis'] == 'keymap'
        assert event['project'] == 'default'
        assert sub.queue.empty()

    import asyncio

    asyncio.run(_run())
    event_hub._HUB = None
