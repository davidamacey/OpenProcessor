"""The beat schedule names exactly the periodic tasks, each routed to ``periodic``, limited, and
with ``expires`` equal to its interval so a backlog behind a stalled consumer is dropped.

https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html
"""

from __future__ import annotations

import pytest

from src.config.queue import TaskQueues
from src.config.task_routes import ROUTES
from src.tasks.beat_schedule import build_schedule


EXPECTED = {
    'reaper',
    'ingest_dispatch',
    'open_vocab_sweep',
    'job_stats',
    'vlm_batch_staging_sweep',
    'upload_sweep',
    'cache_prune',
    'reload_promoted',
    'serving_inventory',
    'cluster_refresh',
    'near_dup_rescore',
    'near_dup_full_build',
}


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


def test_the_schedule_names_exactly_the_periodic_tasks() -> None:
    assert set(build_schedule()) == EXPECTED


def test_every_entry_is_a_registered_task_on_the_periodic_queue_that_expires(worker_app) -> None:
    for name, entry in build_schedule().items():
        task = worker_app.tasks[entry['task']]
        assert ROUTES[entry['task']] == TaskQueues.PERIODIC, name
        assert (task.soft_time_limit, task.time_limit) == (120, 150), name
        assert entry['options'] == {'expires': entry['schedule']}, name


def test_the_app_applies_the_built_schedule(worker_app) -> None:
    assert set(worker_app.conf.beat_schedule) == EXPECTED


def test_the_ingest_dispatch_interval_is_the_env_value_and_zero_turns_it_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_INGEST_DISPATCH_S', '12')
    assert build_schedule()['ingest_dispatch']['schedule'] == 12.0
    monkeypatch.setenv('OP_INGEST_DISPATCH_S', '0')
    assert 'ingest_dispatch' not in build_schedule()


def test_the_cluster_refresh_interval_is_the_env_value_and_zero_turns_it_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert build_schedule()['cluster_refresh']['schedule'] == 120.0
    monkeypatch.setenv('OP_CLUSTER_REFRESH_INTERVAL_S', '45')
    assert build_schedule()['cluster_refresh']['schedule'] == 45.0
    monkeypatch.setenv('OP_CLUSTER_REFRESH_INTERVAL_S', '0')
    assert 'cluster_refresh' not in build_schedule()
