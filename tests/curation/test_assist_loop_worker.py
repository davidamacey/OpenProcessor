"""The loop worker's cycle: pick the oldest queued loop among the projects it serves, bind that
project for the whole run, run it, and repair loops a dead worker left behind. The loop itself
is faked here (runtime.run_job is tested on its own)."""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

import pytest

from scripts.curation import assist_loop_worker as worker
from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project, current_project
from src.config.projects import new_project_record
from src.services.curation.assist_loop import job
from src.services.curation.assist_loop.driver import LoopOutcome
from src.services.curation.file_job import HEARTBEAT_STALE_S


if TYPE_CHECKING:
    from pathlib import Path

    from src.config.projects import ProjectRecord


@pytest.fixture(autouse=True)
def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_ASSIST_LOOP_JOBS_DIR', str(tmp_path / 'loops'))


def _record(slug: str) -> ProjectRecord:
    return new_project_record(slug, base_curation_config())


def _queue(record: ProjectRecord) -> str:
    with bind_project(record):
        return job.create_loop(state={'status': 'queued'}, request={}).directory.name


class Runs:
    """Replaces runtime.run_job: records (project, loop) and ends the loop completed."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ran: list[tuple[str, str]] = []

        async def run_job(_os: Any, loop_id: str) -> LoopOutcome:
            self.ran.append((current_project().record.slug, loop_id))
            job.open_job(loop_id).update(status='completed')
            return LoopOutcome('completed')

        monkeypatch.setattr(worker, 'run_job', run_job)


def _cycle(projects: list[ProjectRecord]) -> bool:
    return asyncio.run(worker.run_next(projects, object()))


def test_a_queued_loop_is_run_with_its_project_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = Runs(monkeypatch)
    alpha = _record('alpha')
    loop_id = _queue(alpha)

    assert _cycle([alpha]) is True

    assert runs.ran == [('alpha', loop_id)]
    assert _cycle([alpha]) is False, 'the trigger was claimed'


def test_nothing_queued_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = Runs(monkeypatch)

    assert _cycle([_record('alpha')]) is False
    assert runs.ran == []


def test_the_oldest_trigger_goes_first_across_projects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = Runs(monkeypatch)
    alpha, beta = _record('alpha'), _record('beta')
    first = _queue(beta)
    time.sleep(0.02)
    second = _queue(alpha)

    _cycle([alpha, beta])
    _cycle([alpha, beta])

    assert runs.ran == [('beta', first), ('alpha', second)]


def test_a_project_the_worker_does_not_serve_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = Runs(monkeypatch)
    alpha, beta = _record('alpha'), _record('beta')
    _queue(beta)

    assert _cycle([alpha]) is False
    assert runs.ran == []
    with bind_project(beta):
        assert job.pending_trigger_mtime() is not None


def test_a_trigger_for_a_cancelled_loop_is_never_run(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = Runs(monkeypatch)
    alpha = _record('alpha')
    with bind_project(alpha):
        loop_id = job.create_loop(state={'status': 'queued'}, request={}).directory.name
        job.cancel_loop(loop_id)

    assert _cycle([alpha]) is False
    assert runs.ran == []


def test_a_bug_in_the_run_marks_the_loop_failed_and_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(_os: Any, _loop_id: str) -> LoopOutcome:
        msg = 'bug in the loop'
        raise KeyError(msg)

    monkeypatch.setattr(worker, 'run_job', boom)
    alpha = _record('alpha')
    loop_id = _queue(alpha)

    with pytest.raises(KeyError):
        _cycle([alpha])

    with bind_project(alpha):
        state = job.open_job(loop_id).read()
    assert state['status'] == 'failed'
    assert 'KeyError' in state['error']
    assert state['finished_at']


def test_startup_repairs_loops_a_dead_worker_left_running_in_every_project() -> None:
    projects = [_record('alpha'), _record('beta')]
    made = []
    for record in projects:
        with bind_project(record):
            loop = job.create_loop(state={'status': 'queued'}, request={})
            loop.update(status='running')
            loop.touch_heartbeat()
            old = time.time() - HEARTBEAT_STALE_S - 5
            os.utime(loop.heartbeat_file, (old, old))
            made.append((record, loop.directory.name))

    worker.repair_orphans(projects)

    for record, loop_id in made:
        with bind_project(record):
            assert job.open_job(loop_id).read()['status'] == 'interrupted'
