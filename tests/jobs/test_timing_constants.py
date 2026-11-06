"""The job lease timing keeps the margins the fence relies on (02b section 3.1, review R-2)."""

from __future__ import annotations

from src.domain import job_timing


def test_the_fence_margin_is_inside_the_lease() -> None:
    assert job_timing.FENCE_MARGIN_S < job_timing.LEASE_TTL_S


def test_four_missed_beats_still_fit_in_the_lease() -> None:
    assert job_timing.HEARTBEAT_TICK_S * 4 < job_timing.LEASE_TTL_S


def test_a_holder_stops_writing_before_a_reaper_can_take_over() -> None:
    # may_write() goes false at TTL - margin; the reaper acts only after TTL.
    assert job_timing.LEASE_TTL_S - job_timing.FENCE_MARGIN_S > job_timing.HEARTBEAT_TICK_S


def test_the_documented_values() -> None:
    assert job_timing.LEASE_TTL_S == 120
    assert job_timing.HEARTBEAT_TICK_S == 15
    assert job_timing.FENCE_MARGIN_S == 30
    assert job_timing.PROGRESS_MIN_INTERVAL_S == 1
