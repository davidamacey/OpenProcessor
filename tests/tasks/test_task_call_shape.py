"""Every task takes the job id and nothing else, the shape ``task_queue.enqueue`` sends.

``enqueue`` publishes ``send_task(name, args=[job_id], task_id=job_id)``, so a worker calls the
task with exactly one positional argument. The rule: every task in ``ROUTES`` (the names
``enqueue`` accepts) must bind exactly one positional job id. Periodic/beat entries that are
not in ``ROUTES`` are not sent by ``enqueue`` and are outside the rule. Celery's testing
guide runs the task object directly (``.apply``/``.run``) and does not use
``task_always_eager`` (https://docs.celeryq.dev/en/stable/userguide/testing.html).
"""

from __future__ import annotations

import inspect

import pytest

from src.config.task_routes import ROUTES


@pytest.fixture
def worker_app(queue_env: None):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


def test_noop_is_the_wrapper_task_of_its_kind(worker_app) -> None:
    """``op.infra.noop`` runs through ``register_job``: the kind's limits, the job id as its argument.

    Running it needs a claimed ``jobs`` row; ``tests/tasks/test_job_task_wrapper.py`` covers that
    against a database.
    """
    from src.config.job_kinds import KINDS

    task = worker_app.tasks['op.infra.noop']
    assert task.soft_time_limit == KINDS['noop'].soft_limit_s
    assert task.time_limit == KINDS['noop'].hard_limit_s
    assert list(inspect.signature(task.run).parameters) == ['job_id']


@pytest.mark.parametrize('name', sorted(ROUTES))
def test_every_routed_task_accepts_exactly_one_job_id(worker_app, name: str) -> None:
    task = worker_app.tasks[name]
    inspect.signature(task.run).bind('job-1')
    with pytest.raises(TypeError):
        inspect.signature(task.run).bind()
    with pytest.raises(TypeError):
        inspect.signature(task.run).bind('job-1', 'extra')
