"""Issue #212: when the periodic cluster refresh starts a run, with a fake clock and fake
observations. Scheduling only: what a run computes is pinned by
``test_cluster_schedule_equivalence.py`` and ``tests/clustering/``. ``built`` is the pool size of
the newest finished partition, so growth is measured against what clustering last saw.
"""

from __future__ import annotations

import pytest

from src.services.clustering.refresh_schedule import (
    Action,
    Observation,
    Policy,
    ProjectSchedule,
    step,
)


POLICY = Policy(threshold=200, max_deferral_s=600.0, retry_backoff_s=300.0)


def obs(
    count: int,
    *,
    built: int = 1000,
    unfinished: int = 0,
    tracked: str | None = None,
    busy: bool = False,
    recent_ingest: bool = False,
):
    return Observation(
        count=count,
        built=built,
        unfinished=unfinished,
        tracked_status=tracked,
        busy=busy,
        loop_live=False,
        recent_ingest=recent_ingest,
    )


def settled(count: int = 1000) -> ProjectSchedule:
    """A project that has seen ``count`` since its last run."""
    return ProjectSchedule(prev_count=count)


def test_below_threshold_does_nothing() -> None:
    s = settled(1000)
    assert step(s, obs(1100), 0.0, POLICY) is Action.NONE
    assert s.pending_since is None


def test_defers_while_ingest_is_growing() -> None:
    s = settled(1000)
    assert step(s, obs(1250), 0.0, POLICY) is Action.DEFER
    assert step(s, obs(1500), 120.0, POLICY) is Action.DEFER
    assert s.pending_since == 0.0


def test_defers_while_regions_are_unfinished_even_if_count_is_flat() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    assert step(s, obs(1250, unfinished=30), 120.0, POLICY) is Action.DEFER


def test_runs_once_ingest_is_quiet() -> None:
    s = settled(1000)
    assert step(s, obs(1250), 0.0, POLICY) is Action.DEFER
    assert step(s, obs(1250), 120.0, POLICY) is Action.START
    s.on_started('job-1')
    assert s.pending_since is None
    assert s.job_id == 'job-1'


def test_max_deferral_is_honoured_under_continuous_ingest() -> None:
    s = settled(1000)
    assert step(s, obs(1250), 0.0, POLICY) is Action.DEFER
    assert step(s, obs(1500), 300.0, POLICY) is Action.DEFER
    assert step(s, obs(1800), 600.0, POLICY) is Action.START


def test_repeated_requests_coalesce_into_one_pending_run() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    step(s, obs(1600), 120.0, POLICY)
    step(s, obs(2100), 240.0, POLICY)
    assert s.pending_since == 0.0  # one request, dated from the first crossing
    assert step(s, obs(2100), 360.0, POLICY) is Action.START
    s.on_started('job-1')
    assert step(s, obs(2100, tracked='running'), 480.0, POLICY) is Action.NONE


def test_never_started_twice_while_in_flight() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    assert step(s, obs(1250), 120.0, POLICY) is Action.START
    s.on_started('job-1')
    # More growth crosses the threshold again while the run is live: the request is kept
    # (coalesced) but nothing starts.
    assert step(s, obs(1500, tracked='running'), 240.0, POLICY) is Action.NONE
    assert step(s, obs(1500, tracked='queued'), 360.0, POLICY) is Action.NONE
    assert s.pending_since is not None
    # The run ends with a partition of 1250; the kept request is still due (1500 - 1250 >= 200)
    # and runs once, when quiet.
    done = obs(1500, built=1250, tracked='completed')
    assert step(s, done, 480.0, POLICY) is Action.START


def test_a_completed_run_clears_the_request_when_its_partition_covers_the_growth() -> None:
    s = settled(1250)
    s.job_id, s.pending_since = 'job-1', 120.0
    assert step(s, obs(1250, built=1250, tracked='completed'), 480.0, POLICY) is Action.NONE
    assert (s.job_id, s.pending_since) == (None, None)


def test_another_runs_presence_in_the_project_blocks_start() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    assert step(s, obs(1250, busy=True), 120.0, POLICY) is Action.NONE
    assert step(s, obs(1250), 240.0, POLICY) is Action.START


def test_a_failed_run_is_retried_after_the_backoff() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    step(s, obs(1250), 120.0, POLICY)
    s.on_started('job-1')
    # The reaper marked a dead worker's run interrupted: no partition came of it.
    assert step(s, obs(1250, tracked='interrupted'), 240.0, POLICY) is Action.NONE
    assert s.job_id is None
    assert s.retry_not_before == 540.0
    assert step(s, obs(1250), 300.0, POLICY) is Action.NONE  # backing off
    assert step(s, obs(1250), 600.0, POLICY) is Action.START


def test_an_unknown_tracked_run_is_treated_as_lost_and_retried() -> None:
    s = settled(1250)
    s.job_id = 'gone'
    assert step(s, obs(1250, tracked='lost'), 0.0, POLICY) is Action.NONE
    assert s.job_id is None
    assert s.retry_not_before == POLICY.retry_backoff_s


def test_a_paused_or_cancelling_run_still_holds_the_project() -> None:
    for status in ('paused', 'cancelling'):
        s = settled(1250)
        s.job_id = 'job-1'
        assert step(s, obs(1500, tracked=status), 0.0, POLICY) is Action.NONE
        assert s.job_id == 'job-1'


def test_first_ever_observation_requests_a_run() -> None:
    s = ProjectSchedule()
    assert step(s, obs(500, built=0), 0.0, POLICY) is Action.START


def test_projects_do_not_share_a_clock() -> None:
    a, b = settled(1000), settled(1000)
    assert step(a, obs(1300), 0.0, POLICY) is Action.DEFER
    assert step(b, obs(1010), 0.0, POLICY) is Action.NONE
    assert step(a, obs(1300), 120.0, POLICY) is Action.START
    a.on_started('ja')
    assert b.job_id is None
    assert b.pending_since is None


def test_empty_project_never_requests_a_run() -> None:
    """#350: zero items with nothing built must not read as 'never clustered'."""
    s = ProjectSchedule()
    for cycle in range(5):
        assert step(s, obs(0, built=0), cycle * 120.0, POLICY) is Action.NONE
    assert s.pending_since is None


def test_idle_project_queues_one_run_then_nothing() -> None:
    s = ProjectSchedule()
    assert step(s, obs(50, built=0), 0.0, POLICY) is Action.START
    s.on_started('job-1')
    for cycle in range(1, 6):
        done = obs(50, built=50, tracked='completed')
        assert step(s, done, cycle * 120.0, POLICY) is Action.NONE


def test_growth_past_the_threshold_from_empty_triggers_exactly_once() -> None:
    s = ProjectSchedule()
    assert step(s, obs(0, built=0), 0.0, POLICY) is Action.NONE
    assert step(s, obs(300, built=0), 120.0, POLICY) is Action.DEFER
    assert step(s, obs(300, built=0), 240.0, POLICY) is Action.START
    s.on_started('job-1')
    for cycle in range(3, 7):
        done = obs(300, built=300, tracked='completed')
        assert step(s, done, cycle * 120.0, POLICY) is Action.NONE


def test_recent_ingest_defers_even_when_count_is_flat_and_nothing_pending() -> None:
    s = settled(1250)
    step(s, obs(1250, recent_ingest=True), 0.0, POLICY)
    assert step(s, obs(1250, recent_ingest=True), 120.0, POLICY) is Action.DEFER


def test_recent_ingest_must_be_given() -> None:
    with pytest.raises(TypeError):
        Observation(  # type: ignore[call-arg]
            count=1, built=0, unfinished=0, tracked_status=None, busy=False, loop_live=False
        )
