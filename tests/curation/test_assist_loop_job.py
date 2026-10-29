"""The loop's job directory protocol: one live loop per project, the trigger the API writes and
the worker claims, resume, and the repair of a loop whose worker died."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

import pytest

from src.core.exceptions import RequestRejectedError
from src.services.curation.assist_loop import job
from src.services.curation.file_job import HEARTBEAT_STALE_S, FileJob


if TYPE_CHECKING:
    from pathlib import Path

REQUEST = {'budget': {'max_vlm_crops': 10}}


@pytest.fixture(autouse=True)
def loops_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('OP_ASSIST_LOOP_JOBS_DIR', str(tmp_path / 'loops'))
    return tmp_path / 'loops'


def _queued() -> FileJob:
    return job.create_loop(state={'status': 'queued'}, request=REQUEST)


def _age_heartbeat(loop: FileJob) -> None:
    old = time.time() - HEARTBEAT_STALE_S - 5
    os.utime(loop.heartbeat_file, (old, old))


def test_a_loop_is_stored_per_project_and_marked(loops_root: Path) -> None:
    loop = _queued()

    assert loop.directory.parent == loops_root / 'projects' / 'default'
    assert (loop.directory.parent / '.project').read_text() == 'default'
    assert loop.directory.name.startswith('alp_')
    assert json.loads((loop.directory / 'request.json').read_text()) == REQUEST
    assert loop.read()['loop_id'] == loop.directory.name
    assert loop.read()['status'] == 'queued'


def test_creating_a_loop_writes_the_trigger_the_worker_claims() -> None:
    loop = _queued()

    assert job.claim_trigger() == loop.directory.name
    assert job.claim_trigger() is None, 'a trigger is claimed once'


def test_a_second_loop_is_refused_while_one_is_live() -> None:
    first = _queued()

    with pytest.raises(RequestRejectedError) as busy:
        _queued()

    assert busy.value.status_code == 409
    assert busy.value.detail['error'] == 'assist_loop_busy'
    assert busy.value.detail['loop_id'] == first.directory.name


def test_a_finished_loop_frees_the_slot() -> None:
    first = _queued()
    first.update(status='completed')

    second = _queued()

    assert second.directory != first.directory
    assert job.live_loop_ids() == [second.directory.name]


@pytest.mark.parametrize(
    ('status', 'live'),
    [
        ('queued', True),
        ('running', True),
        ('paused', False),
        ('completed', False),
        ('cancelled', False),
        ('failed', False),
        ('interrupted', False),
    ],
)
def test_only_a_queued_or_running_loop_is_live(status: str, live: bool) -> None:
    _queued().update(status=status)

    assert job.loop_live() is live


def test_a_loop_whose_worker_died_is_not_live_and_reads_interrupted() -> None:
    loop = _queued()
    loop.update(status='running')
    loop.touch_heartbeat()
    _age_heartbeat(loop)

    assert job.loop_live() is False
    assert job.read_state(loop.directory.name)['status'] == 'interrupted'


def test_startup_repairs_a_loop_a_dead_worker_left_running() -> None:
    loop = _queued()
    loop.update(status='running')
    loop.touch_heartbeat()
    _age_heartbeat(loop)

    assert job.reconcile_orphaned_jobs() is True
    assert loop.read()['status'] == 'interrupted'
    assert job.reconcile_orphaned_jobs() is False


def test_the_hold_refuses_with_409_while_a_loop_is_live() -> None:
    job.require_no_live_loop()
    loop = _queued()

    with pytest.raises(RequestRejectedError) as held:
        job.require_no_live_loop()

    assert held.value.status_code == 409
    assert held.value.detail['error'] == 'assist_loop_running'
    assert held.value.detail['loop_id'] == loop.directory.name
    loop.update(status='completed')
    job.require_no_live_loop()


@pytest.mark.parametrize('status', ['interrupted', 'paused', 'cancelled'])
def test_resume_requeues_a_stopped_loop_and_clears_its_flags(status: str) -> None:
    loop = _queued()
    job.claim_trigger()
    loop.update(status=status, error='x', finished_at='then')
    loop.request_cancel()
    loop.touch_heartbeat()
    (loop.directory / 'pause.flag').touch()

    job.resume_loop(loop.directory.name)

    state = loop.read()
    assert state['status'] == 'queued'
    assert state['error'] is None
    assert not loop.cancel_file.exists()
    assert not (loop.directory / 'pause.flag').exists()
    assert job.claim_trigger() == loop.directory.name


@pytest.mark.parametrize('status', ['queued', 'running', 'completed', 'failed'])
def test_a_loop_in_any_other_state_cannot_be_resumed(status: str) -> None:
    loop = _queued()
    loop.update(status=status)
    loop.touch_heartbeat()

    with pytest.raises(RequestRejectedError) as refused:
        job.resume_loop(loop.directory.name)

    assert refused.value.detail['error'] == 'assist_loop_not_resumable'
    assert loop.read()['status'] == status


def test_resume_is_refused_while_another_loop_is_live() -> None:
    old = _queued()
    old.update(status='paused')
    live = _queued()

    with pytest.raises(RequestRejectedError) as busy:
        job.resume_loop(old.directory.name)

    assert busy.value.detail['error'] == 'assist_loop_busy'
    assert busy.value.detail['loop_id'] == live.directory.name


def test_cancel_touches_the_flag_of_a_live_loop_only() -> None:
    loop = _queued()
    assert job.cancel_loop(loop.directory.name) is True
    assert loop.cancel_requested()

    loop.update(status='completed')
    assert job.cancel_loop(loop.directory.name) is False
    assert job.cancel_loop('alp_unknown') is False


def test_a_loop_id_cannot_name_a_path() -> None:
    assert job.read_state('../x') is None
    assert job.read_state('') is None
    assert job.cancel_loop('a/b') is False


def test_projects_are_listed_that_have_a_pending_trigger() -> None:
    assert job.claim_trigger() is None
    loop = _queued()

    assert job.pending_trigger_mtime() is not None
    job.claim_trigger()
    assert job.pending_trigger_mtime() is None
    assert loop.read()['status'] == 'queued'


def test_cancelling_a_loop_the_worker_has_not_claimed_cancels_it_at_once() -> None:
    loop = _queued()

    assert job.cancel_loop(loop.directory.name) is True

    assert loop.read()['status'] == 'cancelled'
    assert job.claim_trigger() is None, 'the worker must not pick up a cancelled loop'


def test_resuming_an_unknown_loop_is_a_404() -> None:
    with pytest.raises(RequestRejectedError) as missing:
        job.resume_loop('alp_unknown')

    assert missing.value.status_code == 404
