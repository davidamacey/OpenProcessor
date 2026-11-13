"""Only a class a person creates is given a hotkey letter; a detector seed or an import is not.

A 70-class detector seed would otherwise use every free letter on rare classes, leaving none for
``car`` or for the classes the user creates later."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.classes import registry as registry_module
from src.services.classes.registry import ClassRegistry


pytestmark = pytest.mark.asyncio


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[Any, ...]]:
    seen: dict[str, tuple[Any, ...]] = {}

    async def run(self: ClassRegistry, fn: Any) -> Any:
        return await fn(None, 'pid')

    async def letters(session: Any, project_id: Any) -> list[str]:
        return ['x']

    async def add(*args: Any) -> Any:
        seen['add'] = args
        return type('Row', (), {'class_id': 1})()

    async def ensure(*args: Any) -> Any:
        seen['ensure'] = args
        return type('Row', (), {'class_id': 1})(), True

    monkeypatch.setattr(ClassRegistry, '_write', run)
    for name in ('keymap_letters', '_keymap_letters'):
        monkeypatch.setattr(registry_module, name, letters, raising=False)
    monkeypatch.setattr(registry_module.repo, 'add_class', add)
    monkeypatch.setattr(registry_module.repo, 'ensure_class', ensure)
    monkeypatch.setattr(registry_module, 'entry_from_row', lambda row: row)
    return seen


async def test_a_class_a_person_adds_is_offered_the_keymap_letters(calls: dict[str, Any]) -> None:
    await ClassRegistry('p').add_class('car')
    assert calls['add'][-1] == ['x']


async def test_a_seeded_or_imported_class_is_not_given_a_letter(calls: dict[str, Any]) -> None:
    await ClassRegistry('p').ensure_entry('car', group='detector')
    assert ['x'] not in calls['ensure']
    assert len(calls['ensure']) == 5
