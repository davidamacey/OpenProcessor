"""The keymap override map in project_keymaps: optimistic writes, defaults dropped, the event."""

from __future__ import annotations

import asyncio

import pytest
from _config_fixtures import make_project

from src.services.curation import keymap


pytestmark = pytest.mark.db


@pytest.fixture(scope='module', autouse=True)
def _default_project(pg):
    return make_project(pg, 'default')


@pytest.mark.asyncio
async def test_a_project_with_a_default_row_reads_as_the_defaults(pg_env) -> None:
    doc = await keymap.get_keymap_doc()
    assert doc.overrides == {}
    assert doc.is_default is True
    assert doc.revision == 1


@pytest.mark.asyncio
async def test_two_saves_with_one_revision_one_conflicts(pg_env) -> None:
    registry = keymap.load_registry()
    action = next(a for a in registry.actions.values() if a.modifiable)
    start = (await keymap.get_keymap_doc()).revision

    results = await asyncio.gather(
        keymap.save_keymap_doc(overrides={action.id: ['ctrl+k']}, expected_revision=start),
        keymap.save_keymap_doc(overrides={action.id: ['ctrl+j']}, expected_revision=start),
        return_exceptions=True,
    )
    assert sorted(type(r).__name__ for r in results) == ['KeymapDoc', 'RevisionConflictError']
    stored = await keymap.get_keymap_doc()
    assert stored.revision == start + 1
    assert stored.is_default is False


@pytest.mark.asyncio
async def test_an_override_equal_to_the_default_is_dropped(pg_env) -> None:
    registry = keymap.load_registry()
    action = next(a for a in registry.actions.values() if a.modifiable)
    current = await keymap.get_keymap_doc()
    saved = await keymap.save_keymap_doc(
        overrides={action.id: list(action.default)}, expected_revision=current.revision
    )
    assert saved.overrides == {}
    assert saved.is_default is True
