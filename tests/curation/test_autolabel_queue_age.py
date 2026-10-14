"""Issue #236: an auto_label job queued while the worker is down has no heartbeat,
so nothing repaired it and the cluster-refresh daemon waited on it forever. A queued
job now has a bounded age, served in its status; past it the job fails with the reason
and its trigger is withdrawn."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.services.curation.autolabel import job


if TYPE_CHECKING:
    from pathlib import Path


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


def _pipeline() -> None:
    return None


@pytest.fixture
def clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Clock:
    monkeypatch.setattr(job, '_state_dir', lambda: tmp_path / 'auto_label')
    fake = _Clock(1_000_000.0)
    monkeypatch.setattr(job, 'time', fake)
    return fake


def test_a_queued_job_reports_its_age_and_the_limit(clock: _Clock) -> None:
    job.start_job(_pipeline, {})
    clock.now += 120

    state = job.get_state()

    assert state['status'] == 'queued'
    assert state['queue_age_seconds'] == 120
    assert state['queue_max_age_seconds'] == job.QUEUE_MAX_AGE_S


def test_a_job_queued_past_the_limit_fails_with_the_reason(clock: _Clock) -> None:
    started = job.start_job(_pipeline, {})
    clock.now += job.QUEUE_MAX_AGE_S + 1

    state = job.get_state()

    assert state['status'] == 'failed'
    assert state['job_id'] == started['job_id']
    assert 'worker' in state['error']
    assert state['finished_at'] > 0
    assert not job._trigger_file().exists()  # a worker that starts later must not run it


def test_the_failure_is_persisted_and_a_new_job_can_start(clock: _Clock) -> None:
    job.start_job(_pipeline, {})
    clock.now += job.QUEUE_MAX_AGE_S + 1
    job.get_state()

    assert json.loads(job._job().state_file.read_text())['status'] == 'failed'
    again = job.start_job(_pipeline, {})
    assert again['status'] == 'queued'


def test_a_job_just_under_the_limit_is_still_queued(clock: _Clock) -> None:
    job.start_job(_pipeline, {})
    clock.now += job.QUEUE_MAX_AGE_S - 1

    assert job.get_state()['status'] == 'queued'
    assert job._trigger_file().exists()


def test_a_claimed_job_is_not_failed_by_its_queue_age(clock: _Clock) -> None:
    """The worker unlinks the trigger when it claims the job; until it writes
    ``running`` the state still says queued, which is the worker's, not a queue wait."""
    job.start_job(_pipeline, {})
    job._trigger_file().unlink()
    clock.now += job.QUEUE_MAX_AGE_S * 2

    assert job.get_state()['status'] == 'queued'


def test_a_corrupt_state_blocks_a_new_start(clock: _Clock) -> None:
    job._job().directory.mkdir(parents=True)
    job._job().state_file.write_text('{not json')

    with pytest.raises(RuntimeError, match='already in progress'):
        job.start_job(_pipeline, {})
