"""The server-computed drain-stability verdict.

Persisted to a state.json under ``OP_REGION_DRAIN_STATE_DIR`` rather than
kept in module memory, so the verdict is correct regardless of which
``yolo-api --workers`` process answers a given poll -- every test here
points that env var at a per-test ``tmp_path``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from src.services.curation import region_drain


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    d = tmp_path / 'region_drain'
    monkeypatch.setenv('OP_REGION_DRAIN_STATE_DIR', str(d))
    yield d
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


# =============================================================================
# Multi-process persistence -- the actual bug
# =============================================================================


def test_streak_persists_across_a_simulated_second_process(state_dir: Path):
    """The bug: an ingest walker's consecutive polls landing on different
    worker processes used to each see streak=0 (module memory, not
    shared). Simulates a second process by writing the on-disk state
    directly (as process A's poll would have left it) and then calling
    ``observe_drain`` fresh -- as process B's poll would -- with no
    Python object shared between the two steps."""
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / 'state.json').write_text(json.dumps({'streak': 2, 'zero_since': t0.timestamp()}))
    # "Process B" polls next, one interval later, still zero.
    v = region_drain.observe_drain(0, now=t0 + timedelta(seconds=1))
    assert v.drained is True  # streak becomes 3 == default stable_polls
    on_disk = json.loads((state_dir / 'state.json').read_text())
    assert on_disk['streak'] == 3


def test_stable_for_s_grows_across_simulated_processes(state_dir: Path):
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    v1 = region_drain.observe_drain(0, now=t0)
    assert v1.stable_for_s == 0.0

    # A different "process" reads the file fresh, sees the same
    # zero_since, ten seconds later.
    v2 = region_drain.observe_drain(0, now=t0 + timedelta(seconds=10))
    assert v2.stable_for_s == pytest.approx(10.0)

    v3 = region_drain.observe_drain(0, now=t0 + timedelta(seconds=25))
    assert v3.stable_for_s == pytest.approx(25.0)


def test_nonzero_reading_from_a_simulated_process_clears_persisted_streak(state_dir: Path):
    """Work reappearing must clear the persisted streak/zero_since even
    though a *different* on-disk state was written by a "prior process"."""
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / 'state.json').write_text(json.dumps({'streak': 5, 'zero_since': t0.timestamp()}))
    v = region_drain.observe_drain(4, now=t0 + timedelta(seconds=1))
    assert v.drained is False
    assert v.stable_for_s == 0.0
    on_disk = json.loads((state_dir / 'state.json').read_text())
    assert on_disk == {'streak': 0, 'zero_since': None}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
