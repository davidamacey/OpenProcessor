"""Every task takes the job id and nothing else, the shape ``task_queue.enqueue`` sends.

``enqueue`` publishes ``send_task(name, args=[job_id], task_id=job_id)``, so a worker calls the
task with exactly one positional argument. The rule: every job task in ``ROUTES`` (the names
``enqueue`` accepts) must bind exactly one positional job id. Beat ticks, on the non-job queues,
have no job row and take no argument. Celery's testing
guide runs the task object directly (``.apply``/``.run``) and does not use
``task_always_eager`` (https://docs.celeryq.dev/en/stable/userguide/testing.html).
"""

from __future__ import annotations

import inspect

import pytest

from src.config.queue import NON_JOB_QUEUES, TaskQueues
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


# The GPU queues are consumed by their own images, whose modules register these tasks through
# register_job (the same wrapper); the API's worker app never imports them.
GPU_QUEUES = frozenset({TaskQueues.GPU_TRAIN, TaskQueues.GPU_EXPORT, TaskQueues.GPU_EVAL})
JOB_TASKS = sorted(
    name for name, queue in ROUTES.items() if queue not in NON_JOB_QUEUES | GPU_QUEUES
)
BEAT_TASKS = sorted(name for name, queue in ROUTES.items() if queue in NON_JOB_QUEUES)


@pytest.mark.parametrize('name', BEAT_TASKS)
def test_a_beat_task_takes_no_argument(worker_app, name: str) -> None:
    """A beat tick has no job row: beat sends it with no arguments."""
    task = worker_app.tasks[name]
    inspect.signature(task.run).bind()
    with pytest.raises(TypeError):
        inspect.signature(task.run).bind('job-1')


@pytest.mark.parametrize('name', JOB_TASKS)
def test_every_routed_task_accepts_exactly_one_job_id(worker_app, name: str) -> None:
    task = worker_app.tasks[name]
    inspect.signature(task.run).bind('job-1')
    with pytest.raises(TypeError):
        inspect.signature(task.run).bind()
    with pytest.raises(TypeError):
        inspect.signature(task.run).bind('job-1', 'extra')
