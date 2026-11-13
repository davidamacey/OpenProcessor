"""The job lease timing keeps the margins the fence relies on (02b section 3.1, review R-2)."""

from __future__ import annotations

import importlib

import pytest

from src.domain import job_timing
from src.tasks import beat_schedule


_TIMER_VARS = (
    'OP_JOB_LEASE_TTL_S',
    'OP_JOB_HEARTBEAT_TICK_S',
    'OP_JOB_FENCE_MARGIN_S',
    'OP_REAPER_INTERVAL_S',
)


@pytest.fixture
def production_timers(monkeypatch: pytest.MonkeyPatch):
    """The modules as imported with no timer override (the live harness sets short ones)."""
    for name in _TIMER_VARS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(job_timing)
    importlib.reload(beat_schedule)
    yield
    monkeypatch.undo()
    importlib.reload(job_timing)
    importlib.reload(beat_schedule)


def test_the_fence_margin_is_inside_the_lease() -> None:
    assert job_timing.FENCE_MARGIN_S < job_timing.LEASE_TTL_S


def test_four_missed_beats_still_fit_in_the_lease() -> None:
    assert job_timing.HEARTBEAT_TICK_S * 4 < job_timing.LEASE_TTL_S


def test_a_holder_stops_writing_before_a_reaper_can_take_over() -> None:
    # may_write() goes false at TTL - margin; the reaper acts only after TTL.
    assert job_timing.LEASE_TTL_S - job_timing.FENCE_MARGIN_S > job_timing.HEARTBEAT_TICK_S


def test_the_documented_values(production_timers: None) -> None:
    assert job_timing.LEASE_TTL_S == 120
    assert job_timing.HEARTBEAT_TICK_S == 15
    assert job_timing.FENCE_MARGIN_S == 30
    assert job_timing.PROGRESS_MIN_INTERVAL_S == 1
    assert beat_schedule.REAPER_INTERVAL_S == 30


def test_an_override_changes_the_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_JOB_LEASE_TTL_S', '40')
    monkeypatch.setenv('OP_REAPER_INTERVAL_S', '10')
    try:
        assert importlib.reload(job_timing).LEASE_TTL_S == 40
        assert importlib.reload(beat_schedule).REAPER_INTERVAL_S == 10
    finally:
        monkeypatch.undo()
        importlib.reload(job_timing)
        importlib.reload(beat_schedule)
