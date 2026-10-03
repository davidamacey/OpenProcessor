"""P3: ``clone_settings`` -- settings copied, ``classes`` refused on a
non-empty target, the source stays byte-identical (read-only bind)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch, fake_ensure_indexes


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)
    yield
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


@pytest.fixture(autouse=True)
def _noop_ensure_indexes():
    with patch(
        'src.routers.curation._common._ensure_indexes',
        new=AsyncMock(side_effect=fake_ensure_indexes),
    ):
        yield


def _registry_for(client: FakeLifecycleOpenSearch) -> ProjectRegistry:
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)
    return registry


def test_clone_settings_copies_defaults() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='source', display_name='Source'))
    asyncio.run(lifecycle.create_project(client, slug='target', display_name='Target'))
    asyncio.run(registry.ensure_fresh())
    source = registry.get('source')
    target = registry.get('target')
    assert source is not None
    assert target is not None

    from src.config.project_context import bind_project

    with bind_project(source):
        from src.clients.curation_opensearch import update_curation_settings

        asyncio.run(update_curation_settings(client, {'axis_x': 'value_x'}))

    asyncio.run(
        lifecycle.clone_settings(
            client, target_record=target, from_slug='source', axes=['settings_defaults']
        )
    )

    with bind_project(target):
        from src.clients.curation_opensearch import get_curation_settings

        cloned = asyncio.run(get_curation_settings(client))
    assert cloned['defaults'].get('axis_x') == 'value_x'


def test_clone_classes_refused_on_non_empty_target() -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='source', display_name='Source'))
    asyncio.run(lifecycle.create_project(client, slug='target', display_name='Target'))
    asyncio.run(registry.ensure_fresh())
    target = registry.get('target')
    assert target is not None

    from src.config.curation import items_index
    from src.config.project_context import bind_project

    with bind_project(target):
        client.indexes[items_index()] = [{'item_id': 'x'}]  # non-empty target

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            lifecycle.clone_settings(
                client, target_record=target, from_slug='source', axes=['classes']
            )
        )
    assert exc_info.value.detail['error'] == 'target_not_empty'


def test_clone_source_stays_byte_identical() -> None:
    """The source is read under a read-only bind; a write to it must
    raise, proving clone_settings never mutates the source."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='source', display_name='Source'))
    asyncio.run(lifecycle.create_project(client, slug='target', display_name='Target'))
    asyncio.run(registry.ensure_fresh())
    source = registry.get('source')
    target = registry.get('target')
    assert source is not None
    assert target is not None

    asyncio.run(
        lifecycle.clone_settings(
            client, target_record=target, from_slug='source', axes=['settings_defaults']
        )
    )

    from src.config.project_context import bind_project
    from src.services.projects.guard import ProjectReadOnly, check_request

    with bind_project(source, read_only=True), pytest.raises(ProjectReadOnly):
        check_request(
            'PUT',
            f'/{source.resources.indexes[next(iter(source.resources.indexes))]}/_doc/x',
            None,
            registry.snapshot(),
        )


def test_clone_keymap_axis_all_or_nothing_reports_class_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B2 (W2b Opus review): ``clone_settings`` with ``axes=['keymap']``
    validates the source's stored overrides against the *target's* class
    registry, and is all-or-nothing -- any class conflict (or validation
    error) leaves the target's keymap completely unchanged (not even the
    non-conflicting actions copy) and returns a structured conflict
    report. Dropping only the conflicting actions and writing the rest
    used to be able to leave the target with an invalid keymap (a kept
    override colliding with a dropped action's now-reinstated default);
    see the review's B2 finding.

    ``FakeLifecycleOpenSearch`` keeps one flat ``docs`` dict keyed only by
    doc id (ignoring ``index``, see its docstring) -- fine for the
    project-registry docs it was built for, but it would silently alias
    two projects' identically-named ``keymap:default`` doc onto the same
    slot. The keymap read/write calls are stubbed here (per bound
    project) so this test proves the *clone logic's* per-project
    validation and all-or-nothing behavior, not the shared fixture's
    isolation.
    """
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='source', display_name='Source'))
    asyncio.run(lifecycle.create_project(client, slug='target', display_name='Target'))
    asyncio.run(registry.ensure_fresh())
    source = registry.get('source')
    target = registry.get('target')
    assert source is not None
    assert target is not None

    from src.config.project_context import bind_project, try_current_project
    from src.services.curation.keymap import KeymapDoc

    source_doc = KeymapDoc(
        overrides={'cluster.ignore': ['i'], 'review.skip': ['j']},
        revision=1,
        updated_at=None,
        is_default=False,
    )
    target_doc = KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)
    saved: dict[str, dict[str, list[str]]] = {}

    async def _fake_get(_client: Any, _index: str) -> KeymapDoc:
        current = try_current_project()
        return source_doc if current and current.record.slug == 'source' else target_doc

    async def _fake_save(
        _client: Any, _index: str, *, overrides: dict[str, list[str]], expected_revision: int
    ) -> KeymapDoc:
        current = try_current_project()
        saved[current.record.slug if current else '?'] = overrides
        return KeymapDoc(
            overrides=overrides,
            revision=expected_revision + 1,
            updated_at=None,
            is_default=not overrides,
        )

    monkeypatch.setattr('src.services.curation.keymap.get_keymap_doc', _fake_get)
    monkeypatch.setattr('src.services.curation.keymap.save_keymap_doc', _fake_save)

    with bind_project(target):
        from src.routers.curation import get_class_registry

        reg = get_class_registry()
        reg.add_class('ice_cream_truck', group='vehicle')
        loaded = reg.load()
        for c in loaded.classes:
            if c.class_name == 'ice_cream_truck':
                c.hotkey_letter = 'i'
        reg._atomic_write(loaded)

    conflicts = asyncio.run(
        lifecycle.clone_settings(client, target_record=target, from_slug='source', axes=['keymap'])
    )

    # B2: the conflict (combo 'i' is the target's ice_cream_truck hotkey)
    # makes the whole axis a no-op -- 'review.skip' does NOT copy either,
    # and the target's keymap is never written at all.
    assert 'target' not in saved
    assert conflicts == [
        {
            'action_id': 'cluster.ignore',
            'combo': 'i',
            'class_id': 0,
            'class_name': 'ice_cream_truck',
        }
    ]


def test_clone_keymap_axis_copies_overrides_when_no_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clean counterpart to the all-or-nothing test above: with no
    class conflict, the source's overrides copy to the target verbatim."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='source', display_name='Source'))
    asyncio.run(lifecycle.create_project(client, slug='target', display_name='Target'))
    asyncio.run(registry.ensure_fresh())
    source = registry.get('source')
    target = registry.get('target')
    assert source is not None
    assert target is not None

    from src.config.project_context import try_current_project
    from src.services.curation.keymap import KeymapDoc

    source_doc = KeymapDoc(
        overrides={'cluster.ignore': ['i'], 'review.skip': ['j']},
        revision=1,
        updated_at=None,
        is_default=False,
    )
    target_doc = KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)
    saved: dict[str, dict[str, list[str]]] = {}

    async def _fake_get(_client: Any, _index: str) -> KeymapDoc:
        current = try_current_project()
        return source_doc if current and current.record.slug == 'source' else target_doc

    async def _fake_save(
        _client: Any, _index: str, *, overrides: dict[str, list[str]], expected_revision: int
    ) -> KeymapDoc:
        current = try_current_project()
        saved[current.record.slug if current else '?'] = overrides
        return KeymapDoc(
            overrides=overrides,
            revision=expected_revision + 1,
            updated_at=None,
            is_default=not overrides,
        )

    monkeypatch.setattr('src.services.curation.keymap.get_keymap_doc', _fake_get)
    monkeypatch.setattr('src.services.curation.keymap.save_keymap_doc', _fake_save)

    conflicts = asyncio.run(
        lifecycle.clone_settings(client, target_record=target, from_slug='source', axes=['keymap'])
    )

    assert saved['target'] == {'cluster.ignore': ['i'], 'review.skip': ['j']}
    assert conflicts == []


def _clone_keymap(
    monkeypatch: pytest.MonkeyPatch, source_overrides: dict[str, list[str]]
) -> tuple[
    list[dict[str, list[str]]], list[dict[str, Any]], list[tuple[str, dict[str, Any]]], list
]:
    """Clone only the keymap axis with the keymap read/write stubbed.
    Returns ``(saves, published events, warning log calls, conflicts)``."""
    from src.config.project_context import try_current_project
    from src.services.curation import event_hub
    from src.services.curation.keymap import KeymapDoc
    from src.services.projects import clone as clone_mod

    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(lifecycle.create_project(client, slug='source', display_name='Source'))
    asyncio.run(lifecycle.create_project(client, slug='target', display_name='Target'))
    asyncio.run(registry.ensure_fresh())
    target = registry.get('target')
    assert target is not None

    async def _fake_get(_client: Any, _index: str) -> KeymapDoc:
        current = try_current_project()
        if current and current.record.slug == 'source':
            return KeymapDoc(
                overrides=source_overrides, revision=1, updated_at=None, is_default=False
            )
        return KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)

    saves: list[dict[str, list[str]]] = []

    async def _fake_save(
        _client: Any, _index: str, *, overrides: dict[str, list[str]], expected_revision: int
    ) -> KeymapDoc:
        saves.append(overrides)
        return KeymapDoc(
            overrides=overrides, revision=expected_revision + 1, updated_at=None, is_default=False
        )

    events: list[dict[str, Any]] = []

    class _Hub:
        def publish(self, event: dict[str, Any]) -> None:
            events.append(event)

    warnings: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr('src.services.curation.keymap.get_keymap_doc', _fake_get)
    monkeypatch.setattr('src.services.curation.keymap.save_keymap_doc', _fake_save)
    monkeypatch.setattr(event_hub, 'get_event_hub', lambda: _Hub())
    monkeypatch.setattr(
        clone_mod.logger, 'warning', lambda event, **kw: warnings.append((event, kw))
    )
    conflicts = asyncio.run(
        lifecycle.clone_settings(client, target_record=target, from_slug='source', axes=['keymap'])
    )
    return saves, events, warnings, conflicts


def test_clone_keymap_event_carries_both_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    saves, events, _warnings, conflicts = _clone_keymap(monkeypatch, {'cluster.ignore': ['i']})
    assert saves == [{'cluster.ignore': ['i']}]
    assert conflicts == []
    (event,) = [e for e in events if e.get('axis') == 'keymap']
    assert event['keymap_revision'] == 1
    assert 'config_revision' in event


def test_clone_keymap_with_an_invalid_source_keymap_writes_nothing_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``not report.ok`` half of the gate: no class conflict, but the
    source keymap does not validate in the target."""
    saves, events, warnings, conflicts = _clone_keymap(monkeypatch, {'no.such.action': ['i']})
    assert saves == []
    assert conflicts == []
    assert not [e for e in events if e.get('axis') == 'keymap']
    ((event, fields),) = warnings
    assert event == 'keymap_clone_conflicts_left_unchanged'
    assert fields['errors'], 'the refusal must name the validation errors'
