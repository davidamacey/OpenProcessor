"""2E-3: delete_sources runs twice with the second a no-op, and removes exactly what was migrated."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pytest

from src.services.upgrade.from_060.delete_sources import delete_sources
from src.services.upgrade.from_060.export import export_bundle


if TYPE_CHECKING:
    from tests.upgrade._v060 import V060Install


NOW = dt.datetime(2026, 2, 1, tzinfo=dt.UTC)


@pytest.mark.asyncio
async def test_deleting_twice_is_a_no_op_the_second_time(v060: V060Install, tmp_path) -> None:
    await export_bundle(v060.reader, v060.layout, v060.store, tmp_path / 'b', now=NOW)
    first = await delete_sources(tmp_path / 'b', v060.writer, v060.layout)
    second = await delete_sources(tmp_path / 'b', v060.writer, v060.layout)
    assert 'op_projects' in first.indexes_deleted
    assert first.files_deleted > 0
    assert second.to_json() == {
        'indexes_deleted': [],
        'registries_renamed': [],
        'files_deleted': 0,
        'directories_removed': 0,
    }


@pytest.mark.asyncio
async def test_what_stays_is_the_live_data_and_the_artifacts(v060: V060Install, tmp_path) -> None:
    await export_bundle(v060.reader, v060.layout, v060.store, tmp_path / 'b', now=NOW)
    await delete_sources(tmp_path / 'b', v060.writer, v060.layout)
    assert {'op_prj_demo__items', 'op_prj_demo__images', 'op_prj_default__items'} <= set(
        v060.os.indexes
    )
    assert not [
        n for n in v060.os.indexes if n.rsplit('__', 1)[-1] in {'configs', 'settings', 'classes'}
    ]
    assert v060.path('data/projects/demo/exports/export_1/manifest.json').is_file()  # an artifact
    assert v060.path('models/demo__car_detector/promote.json').is_file()  # serving state
    assert v060.path('data/projects/default/class_registry.json.migrated-0.7.0').is_file()
    assert not v060.path('state/projects/demo/label_ops').exists()
    assert not v060.path('jobs/imports/projects/demo').exists()
