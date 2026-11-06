"""``op.periodic.reaper``: routed to ``periodic``, limited, scheduled every 30 s, calls one tick."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from src.config.queue import TaskQueues
from src.config.task_routes import ROUTES
from src.services.jobs.reaper import Reaper, ReaperError


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


def test_the_task_is_routed_to_the_periodic_queue_with_its_limits(worker_app) -> None:
    assert ROUTES['op.periodic.reaper'] == TaskQueues.PERIODIC
    task = worker_app.tasks['op.periodic.reaper']
    assert (task.soft_time_limit, task.time_limit) == (120, 150)


def test_beat_schedules_the_reaper_every_30_seconds_and_drops_a_backlog(worker_app) -> None:
    entry = worker_app.conf.beat_schedule['reaper']
    assert entry['task'] == 'op.periodic.reaper'
    assert entry['schedule'] == 30.0
    assert entry['options'] == {'expires': 30}
    assert entry['task'] in worker_app.tasks


def test_running_the_task_runs_one_tick(worker_app, monkeypatch) -> None:
    ticks: list[int] = []
    monkeypatch.setattr('src.tasks.reaper.reap', lambda: ticks.append(1))
    worker_app.tasks['op.periodic.reaper'].apply()
    assert ticks == [1]


def test_with_the_database_down_the_tick_fails_and_every_rule_was_tried() -> None:
    attempts: list[str] = []

    class Down:
        def __enter__(self):
            attempts.append('connect')
            raise OperationalError('select 1', {}, OSError('database is down'))

        def __exit__(self, *_exc):
            return False

    with pytest.raises(ReaperError):
        Reaper(session_factory=Down).tick()
    # rules a, b, c, d, h, g and f each tried (e and i skip while their job kinds are unregistered)
    assert len(attempts) == 7
