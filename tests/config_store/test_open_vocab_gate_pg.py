"""The open-vocabulary gate's hit-rate windows live in ``project_state``."""

from __future__ import annotations

import pytest
from _config_fixtures import make_project

from src.services.curation.open_vocab_gate import load_tracker, save_tracker
from src.services.detection.segmenter_gate import HitRateTracker


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture(scope='module', autouse=True)
def _default_project(pg):
    return make_project(pg, 'default')


@pytest.mark.asyncio
async def test_the_hit_rate_windows_persist_across_passes(pg_env) -> None:
    assert (await load_tracker()).dump() == {}
    tracker = HitRateTracker()
    for hit in (False, False, True):
        tracker.record('cone|cone', hit=hit, window=5)
    await save_tracker(tracker)
    tracker.record('cone|cone', hit=True, window=5)
    await save_tracker(tracker)  # the second pass writes over the stored row

    assert (await load_tracker()).dump() == {'cone|cone': [False, False, True, True]}
