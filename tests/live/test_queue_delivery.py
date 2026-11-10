"""Delivery on every queue the main compose routes (audit: queue delivery was MISSING).

One job per queue, started through the job service, must be claimed and finished by the worker that
consumes that queue and by no other. ``job_attempts.owner`` is ``<container hostname>:<pid>`` of the
claiming worker, so the check compares it with the hostname of the expected service. The beat tick
queues (``periodic``, ``arbiter``) have no job row: a task sent to the queue must be reported
``succeeded`` in the log of its consumer.

Needs the live harness with every consumer of ``docker/test/compose.yml`` up::

    docker compose -p op-live-a -f docker/test/compose.yml up -d --wait
    .venv/bin/python -m pytest tests/live/test_queue_delivery.py -v --no-cov -m live
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.config.job_kinds import KINDS
from src.config.queue import NON_JOB_QUEUES, TaskQueues
from tests.live import _compose
from tests.live._jobs import send_task, start_job_in_project
from tests.live.conftest import wait_until


pytestmark = pytest.mark.live

TERMINAL = {'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted'}

# queue -> (compose service that consumes it, the job kind to start on it, its params)
JOB_QUEUES: dict[str, tuple[str, str, dict[str, Any]]] = {
    TaskQueues.JOBS: ('jobs-worker', 'noop', {}),
    TaskQueues.OPEN_VOCAB: ('open-vocab-worker', 'open_vocab_drain', {}),
    TaskQueues.AUTOLABEL: ('autolabel-worker', 'autolabel', {}),
    TaskQueues.GPU_EXPORT: ('export-worker', 'model_export', {'triton_name': 'delivery_probe'}),
    TaskQueues.GPU_EVAL: ('eval-worker', 'bakeoff', {}),
}
# These three need real inputs (a cluster run, a training set, an assist-loop project); a job on
# each is claimed and finished by its consumer in test_live_clusters.py, test_live_train.py and
# test_live_assist_loop.py.
PROVED_ELSEWHERE = (TaskQueues.ASSIST_LOOP, TaskQueues.CLUSTER, TaskQueues.GPU_TRAIN)
TICK_QUEUES: dict[str, tuple[str, str]] = {
    TaskQueues.PERIODIC: ('periodic-worker', 'op.periodic.job_stats'),
    TaskQueues.ARBITER: ('arbiter-worker', 'op.periodic.gpu_arbiter'),
}


def test_every_queue_is_covered_by_this_file_or_a_named_one() -> None:
    assert set(JOB_QUEUES) | set(TICK_QUEUES) | set(PROVED_ELSEWHERE) == set(TaskQueues.ALL)
    assert set(TICK_QUEUES) == NON_JOB_QUEUES


def _hostname(service: str) -> str:
    return _compose.compose('exec', '-T', service, 'hostname').stdout.strip()


@pytest.mark.parametrize('queue', sorted(JOB_QUEUES))
def test_a_job_on_each_queue_is_claimed_and_finished_by_its_own_worker(queue: str) -> None:
    service, kind, params = JOB_QUEUES[queue]
    assert KINDS[kind].queue == queue
    job_id = start_job_in_project(kind, **params)
    # Project jobs are not in the deployment-wide GET /jobs, so the row is read from the database.
    row = wait_until(
        lambda: (
            _compose.psql(f"SELECT status FROM jobs WHERE id = '{job_id}'") in TERMINAL
            and _compose.psql(
                f"SELECT status, attempt, coalesce(result->>'queue', '') FROM jobs WHERE id = '{job_id}'"
            )
        ),
        timeout=90,
        interval=1.0,
    )
    assert row, f'job {job_id} did not finish on queue {queue}'
    status, attempt, result_queue = row.split('|')
    assert (status, attempt) == ('completed', '1'), row
    owners = _compose.psql(f"SELECT owner FROM job_attempts WHERE job_id = '{job_id}'")
    assert owners.split(':')[0] == _hostname(service), (
        f'queue {queue}: claimed by {owners}, expected a {service} container'
    )
    if queue != TaskQueues.JOBS:
        assert result_queue == queue  # the fake handler of that queue ran it


@pytest.mark.parametrize('queue', sorted(TICK_QUEUES))
def test_a_tick_sent_to_each_beat_queue_runs_on_its_consumer(queue: str) -> None:
    service, task = TICK_QUEUES[queue]
    task_id = str(uuid.uuid4())
    send_task(task, task_id)
    needle = f'{task}[{task_id}] succeeded'
    seen = wait_until(
        lambda: needle in _compose.compose('logs', '--no-log-prefix', service).stdout,
        timeout=60,
        interval=1.0,
    )
    assert seen, f'{service} never reported {needle!r}'
    others = [s for q, (s, _) in TICK_QUEUES.items() if q != queue]
    for other in others:
        assert task_id not in _compose.compose('logs', '--no-log-prefix', other).stdout
