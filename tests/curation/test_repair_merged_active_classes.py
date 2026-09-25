"""F-56 repair script: re-deprecate registry classes left active with
``merged_into`` set (a broken/hand-edited registry, or one written before
this fix existed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from src.clients.curation_opensearch import ClassRegistry


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / 'scripts' / 'curation' / 'repair_merged_active_classes.py'
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location('_repair_merged_active_classes', _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def repair() -> Any:
    return _load_script()


@pytest.fixture
def registry(tmp_path: Any) -> ClassRegistry:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('sedan', group='vehicle')  # 0
    reg.add_class('suv', group='vehicle')  # 1
    reg.add_class('coupe', group='vehicle')  # 2
    return reg


def _entry(registry: ClassRegistry, class_id: int) -> Any:
    entry = registry.get(class_id)
    assert entry is not None
    return entry


def _break_registry(registry: ClassRegistry, class_id: int, merged_into: int) -> None:
    """Simulate a broken registry: merged_into set but deprecated left False
    (a real merge always sets both -- this reproduces a hand-edit / stale
    snapshot / pre-fix write)."""
    reg = registry.load()
    for c in reg.classes:
        if c.class_id == class_id:
            c.merged_into = merged_into
            c.deprecated = False
    registry._atomic_write(reg)


class TestFindBroken:
    def test_finds_active_class_with_merged_into_set(
        self, repair: Any, registry: ClassRegistry
    ) -> None:
        _break_registry(registry, 0, merged_into=1)
        broken = list(repair.find_broken(registry))
        assert [c.class_id for c in broken] == [0]

    def test_ignores_properly_deprecated_merge(self, repair: Any, registry: ClassRegistry) -> None:
        registry.merge_class(source_id=0, target_id=1)  # sets deprecated=True too
        broken = list(repair.find_broken(registry))
        assert broken == []

    def test_ignores_plain_deprecated_class_with_no_merge_target(
        self, repair: Any, registry: ClassRegistry
    ) -> None:
        registry.set_deprecated(0, True)
        broken = list(repair.find_broken(registry))
        assert broken == []

    def test_ignores_active_class_with_no_merge_target(
        self, repair: Any, registry: ClassRegistry
    ) -> None:
        broken = list(repair.find_broken(registry))
        assert broken == []


class TestRun:
    def test_dry_run_reports_but_does_not_write(self, repair: Any, registry: ClassRegistry) -> None:
        _break_registry(registry, 0, merged_into=1)
        rc = repair.run(registry, apply=False, verbose=True)
        assert rc == 0
        assert _entry(registry, 0).deprecated is False  # untouched

    def test_apply_re_deprecates_broken_classes(self, repair: Any, registry: ClassRegistry) -> None:
        _break_registry(registry, 0, merged_into=1)
        _break_registry(registry, 2, merged_into=1)
        rc = repair.run(registry, apply=True, verbose=False)
        assert rc == 0
        assert _entry(registry, 0).deprecated is True
        assert _entry(registry, 0).merged_into == 1
        assert _entry(registry, 2).deprecated is True
        assert _entry(registry, 2).merged_into == 1

    def test_apply_is_idempotent(self, repair: Any, registry: ClassRegistry) -> None:
        _break_registry(registry, 0, merged_into=1)
        assert repair.run(registry, apply=True, verbose=False) == 0
        # Second pass finds nothing left broken.
        assert repair.find_broken(registry) == []
        assert repair.run(registry, apply=True, verbose=False) == 0

    def test_apply_with_no_broken_classes_is_a_noop(
        self, repair: Any, registry: ClassRegistry
    ) -> None:
        assert repair.run(registry, apply=True, verbose=False) == 0
        assert _entry(registry, 0).deprecated is False
        assert _entry(registry, 1).deprecated is False
        assert _entry(registry, 2).deprecated is False
