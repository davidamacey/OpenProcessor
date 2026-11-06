"""2E-3: a 0.6.0 install with a leftover trigger.json and running.lock ends with neither."""

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
async def test_the_auto_label_signal_files_are_gone_after_delete_sources(
    v060: V060Install, tmp_path
) -> None:
    folder = v060.path('jobs/auto_label/projects/demo')
    assert (folder / 'trigger.json').is_file()
    assert (folder / 'running.lock').is_file()

    await export_bundle(v060.reader, v060.layout, v060.store, tmp_path / 'b', now=NOW)
    await delete_sources(tmp_path / 'b', v060.writer, v060.layout)

    assert not (folder / 'trigger.json').exists()
    assert not (folder / 'running.lock').exists()
    assert not folder.exists()  # and the directory they left empty
