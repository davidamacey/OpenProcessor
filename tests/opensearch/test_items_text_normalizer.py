"""The items index carries the compact text normalizer (2F-6, research plate text 5.1): in the
creation body of a new index, and added to every upgraded index by step 3a of the upgrade."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.clients.curation_opensearch.bodies_core import _items_body
from src.clients.curation_opensearch.lifecycle import INDEX_BODIES
from src.config.curation_roles import IndexRole
from src.services.upgrade.from_060 import analysis_settings
from tests.upgrade._v060 import install


if TYPE_CHECKING:
    from pathlib import Path


def test_a_new_items_body_carries_the_normalizer() -> None:
    analysis = _items_body()['settings']['index']['analysis']

    assert analysis['char_filter']['op_alnum_only'] == {
        'type': 'pattern_replace',
        'pattern': '[^\\p{L}\\p{Nd}]',
        'replacement': '',
    }
    assert analysis['normalizer']['op_text_compact'] == {
        'type': 'custom',
        'char_filter': ['op_alnum_only'],
        'filter': ['asciifolding', 'uppercase'],
    }


def test_only_the_items_body_has_analysis_settings() -> None:
    with_analysis = [r for r, b in INDEX_BODIES.items() if 'analysis' in b['settings']['index']]

    assert with_analysis == [IndexRole.ITEMS]


@pytest.fixture
def fixture_items(tmp_path: Path):
    world = install(tmp_path / 'v060')
    names = sorted(n for n in world.os.indexes if n.endswith('__items'))
    assert len(names) >= 2
    return world, names


@pytest.mark.asyncio
async def test_the_step_closes_updates_and_reopens_each_items_index_once(fixture_items) -> None:
    world, names = fixture_items
    lines: list[str] = []

    report = await analysis_settings.apply(world.os, names, say=lines.append)

    assert report.changed == names
    for name in names:
        assert world.os.indexes[name]['settings']['index']['analysis']['normalizer']
    assert world.os.calls.count('indices.close') == len(names)
    assert world.os.calls.count('indices.put_settings') == len(names)
    assert world.os.calls.count('indices.open') == len(names)
    assert world.os.closed == set()
    assert len(lines) == len(names)
    for n, (line, name) in enumerate(zip(lines, names, strict=True), start=1):
        assert line.startswith(f'{n}/{len(names)} {name} closed ')
        assert line.endswith(', reopened, health green')


@pytest.mark.asyncio
async def test_a_second_run_issues_no_write(fixture_items) -> None:
    world, names = fixture_items
    await analysis_settings.apply(world.os, names)
    world.os.calls.clear()

    report = await analysis_settings.apply(world.os, names)

    assert report.changed == []
    assert report.skipped == names
    writes = {'indices.close', 'indices.put_settings', 'indices.open'}
    assert not writes & set(world.os.calls)


@pytest.mark.asyncio
async def test_the_dry_run_lists_every_index_and_writes_nothing(fixture_items) -> None:
    world, names = fixture_items
    lines: list[str] = []

    report = await analysis_settings.dry_run(world.os, names, say=lines.append)

    assert report.changed == names
    assert len(lines) == len(names)
    assert all('planning estimate' in line for line in lines)
    writes = {'indices.close', 'indices.put_settings', 'indices.open'}
    assert not writes & set(world.os.calls)
    assert all('analysis' not in world.os.indexes[n]['settings'].get('index', {}) for n in names)
