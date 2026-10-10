"""WP-1.5 (#212): when the cluster retrain starts, with a fake clock and fake
ingest-status observations. Scheduling only: what a retrain computes is
pinned by ``test_cluster_schedule_equivalence.py``."""

from __future__ import annotations

from src.services.curation.clustering.refresh_schedule import (
    Action,
    Observation,
    Policy,
    ProjectSchedule,
    ScheduleBook,
    step,
)


POLICY = Policy(threshold=200, max_deferral_s=600.0, retry_backoff_s=300.0)


def obs(count: int, *, unfinished: int = 0, tracked: str | None = None, busy: bool = False):
    return Observation(count=count, unfinished=unfinished, tracked_status=tracked, busy=busy)


def settled(count: int = 1000) -> ProjectSchedule:
    """A project that trained once at ``count`` and has seen it since."""
    return ProjectSchedule(last_trained_count=count, prev_count=count)


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
    s.on_started(1250, 'job-1', previous_trained=1000)
    assert s.pending_since is None
    assert s.job_id == 'job-1'
    assert s.last_trained_count == 1250


def test_max_deferral_is_honoured_under_continuous_ingest() -> None:
    s = settled(1000)
    assert step(s, obs(1250), 0.0, POLICY) is Action.DEFER
    assert step(s, obs(1500), 300.0, POLICY) is Action.DEFER
    assert step(s, obs(1800), 600.0, POLICY) is Action.START


def test_repeated_requests_coalesce_into_one_pending_job() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    step(s, obs(1600), 120.0, POLICY)
    step(s, obs(2100), 240.0, POLICY)
    assert s.pending_since == 0.0  # one request, dated from the first crossing
    assert step(s, obs(2100), 360.0, POLICY) is Action.START
    s.on_started(2100, 'job-1', previous_trained=1000)
    assert step(s, obs(2100, tracked='running'), 480.0, POLICY) is Action.NONE


def test_never_started_twice_while_in_flight() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    assert step(s, obs(1250), 120.0, POLICY) is Action.START
    s.on_started(1250, 'job-1', previous_trained=1000)
    # More growth crosses the threshold again while the job runs: the request
    # is kept (coalesced) but nothing starts.
    assert step(s, obs(1500, tracked='running'), 240.0, POLICY) is Action.NONE
    assert step(s, obs(1500, tracked='queued'), 360.0, POLICY) is Action.NONE
    assert s.pending_since is not None
    # The job ends; the kept request runs once, when quiet.
    assert step(s, obs(1500, tracked='completed'), 480.0, POLICY) is Action.START


def test_another_jobs_run_in_the_project_blocks_start() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    assert step(s, obs(1250, busy=True), 120.0, POLICY) is Action.NONE
    assert step(s, obs(1250), 240.0, POLICY) is Action.START


def test_stale_or_failed_job_is_repaired_by_a_retry_after_backoff() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    step(s, obs(1250), 120.0, POLICY)
    s.on_started(1250, 'job-1', previous_trained=1000)
    # The server repaired a dead worker's job to 'interrupted'.
    assert step(s, obs(1250, tracked='interrupted'), 240.0, POLICY) is Action.NONE
    assert s.job_id is None
    assert s.last_trained_count == 1000
    assert step(s, obs(1250), 300.0, POLICY) is Action.NONE  # backing off
    assert step(s, obs(1250), 600.0, POLICY) is Action.START


def test_unknown_tracked_job_is_treated_as_lost_and_retried() -> None:
    s = settled(1000)
    s.job_id, s.trained_before = 'gone', 1000
    s.last_trained_count = 1250
    assert step(s, obs(1250, tracked='unknown'), 0.0, POLICY) is Action.NONE
    assert s.last_trained_count == 1000


def test_first_ever_observation_requests_a_train() -> None:
    s = ProjectSchedule()
    assert step(s, obs(500), 0.0, POLICY) is Action.START


def test_cancel_restores_the_trained_mark_so_it_reruns_after_resume() -> None:
    s = settled(1000)
    step(s, obs(1250), 0.0, POLICY)
    step(s, obs(1250), 120.0, POLICY)
    s.on_started(1250, 'job-1', previous_trained=1000)
    assert s.job_id == 'job-1'
    s.on_cancelled()
    assert s.job_id is None
    assert s.last_trained_count == 1000
    assert step(s, obs(1250), 1000.0, POLICY) is Action.START  # crossed again and quiet


def test_projects_are_isolated() -> None:
    book = ScheduleBook()
    a, b = book.get('alpha'), book.get('beta')
    a.last_trained_count = a.prev_count = 1000
    b.last_trained_count = b.prev_count = 1000
    assert step(a, obs(1300), 0.0, POLICY) is Action.DEFER
    assert step(b, obs(1010), 0.0, POLICY) is Action.NONE
    assert step(a, obs(1300), 120.0, POLICY) is Action.START
    a.on_started(1300, 'ja', previous_trained=1000)
    assert b.job_id is None
    assert b.pending_since is None
    assert book.get('alpha') is a


def test_book_forget_drops_state() -> None:
    book = ScheduleBook()
    book.get('alpha').job_id = 'x'
    book.forget('alpha')
    assert book.get('alpha').job_id is None
