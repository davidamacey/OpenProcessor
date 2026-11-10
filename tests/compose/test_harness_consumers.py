"""The live harness consumes every queue a live check starts a job on.

A job on a queue with no consumer stays `queued` forever, and a queued `cluster_run` makes the
assist loop refuse to start (`clustering_running`). Docs:
https://docs.celeryq.dev/en/stable/reference/cli.html#celery-worker (``-Q``).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from compose.test_consumers_after_b2 import consumers
from src.config.queue import TaskQueues


HARNESS = Path(__file__).resolve().parents[2] / 'docker' / 'test' / 'compose.yml'

# The queues the live checks reach: the job routes (jobs), the beat tasks (periodic), the assist
# loop, the region and cluster refresh (cluster) and training (the fake trainer).
NEEDED = (
    TaskQueues.JOBS,
    TaskQueues.PERIODIC,
    TaskQueues.ASSIST_LOOP,
    TaskQueues.CLUSTER,
    TaskQueues.GPU_TRAIN,
)


def test_the_harness_has_a_consumer_for_every_queue_the_live_checks_use() -> None:
    services = yaml.safe_load(HARNESS.read_text(encoding='utf-8'))['services']
    found = consumers(services)
    assert [q for q in NEEDED if q not in found] == []
