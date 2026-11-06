"""The wrapper binds ``project`` and ``attempt`` next to the ``job_id`` and ``kind`` of the signal.

``structlog``'s ``merge_contextvars`` processor, first in the chain of ``src/core/logging.py``,
puts every bound variable on each log line, so a handler's log carries them. ``task_postrun``
clears them (1B-9). A task with no job row (a beat tick) never binds a project. The task is run
with ``Task.apply`` so Celery sends the real ``task_prerun`` and ``task_postrun`` signals; this
is Celery's documented way to test a task without a broker
(https://docs.celeryq.dev/en/stable/userguide/testing.html).
"""

from __future__ import annotations

import pytest
import structlog
from fixtures.job_rows import make_project, start

from src.db.engine import get_sync_engine
from src.tasks.runtime.register import register_job


pytestmark = pytest.mark.db


@pytest.fixture
def worker_app(monkeypatch, tmp_path):
    password = tmp_path / 'queue_password'
    password.write_text('test-password\n')
    monkeypatch.setenv('OP_QUEUE_URL', 'redis://valkey:6379/0')
    monkeypatch.setenv('OP_QUEUE_PASSWORD_FILE', str(password))
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


def test_a_handlers_log_line_carries_job_id_kind_project_and_attempt(
    pg, job_env, wakes, kinds, worker_app
) -> None:
    project = make_project(pg, 'logproj')
    job = start(get_sync_engine(), project, 't_plain')
    seen: dict = {}

    async def handler(ctx):
        seen.update(structlog.contextvars.get_contextvars())

    register_job('t_plain', handler)
    worker_app.tasks['op.test.plain'].apply(args=[str(job.id)], task_id=str(job.id))
    assert seen['job_id'] == str(job.id)
    assert seen['kind'] == 'op.test.plain'
    assert seen['project'] == 'logproj'
    assert seen['attempt'] == 1
    assert structlog.contextvars.get_contextvars() == {}  # task_postrun cleared it


def test_a_periodic_task_with_no_job_row_logs_without_a_project(worker_app) -> None:
    seen: dict = {}

    @worker_app.task(name='op.test.periodic_probe', shared=False)
    def tick(job_id: str) -> None:
        seen.update(structlog.contextvars.get_contextvars())

    tick.apply(args=['tick-1'], task_id='tick-1')
    assert seen['job_id'] == 'tick-1'
    assert 'project' not in seen
    assert 'attempt' not in seen
