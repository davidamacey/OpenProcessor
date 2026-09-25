"""BA-3: the server-computed drain-stability verdict."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.services.curation import region_drain


@pytest.fixture(autouse=True)
def _reset():
    region_drain._reset_for_tests()
    yield
    region_drain._reset_for_tests()


def test_nonzero_total_is_never_drained():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    v = region_drain.observe_drain(5, now=t0)
    assert v.drained is False
    assert v.stable_for_s == 0.0


def test_a_single_zero_reading_is_not_yet_drained():
    """A single zero right after a burst finishes could be a race, not a
    truly drained queue -- only stable_polls consecutive zeros count."""
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    v = region_drain.observe_drain(0, now=t0)
    assert v.drained is False


def test_drained_after_stable_polls_consecutive_zeros():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(region_drain.region_drain_stable_polls() - 1):
        v = region_drain.observe_drain(0, now=t0 + timedelta(seconds=i))
        assert v.drained is False
    v = region_drain.observe_drain(
        0, now=t0 + timedelta(seconds=region_drain.region_drain_stable_polls())
    )
    assert v.drained is True
    assert v.stable_for_s > 0


def test_a_nonzero_reading_resets_the_streak():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(region_drain.region_drain_stable_polls()):
        region_drain.observe_drain(0, now=t0 + timedelta(seconds=i))
    # Work resumed -- the streak must reset, not stay drained.
    v = region_drain.observe_drain(3, now=t0 + timedelta(seconds=100))
    assert v.drained is False
    assert v.stable_for_s == 0.0
    # And a fresh zero starts a new streak from zero, not resuming the old one.
    v = region_drain.observe_drain(0, now=t0 + timedelta(seconds=101))
    assert v.drained is False
