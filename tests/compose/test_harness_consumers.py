"""The live harness consumes every queue the main compose routes.

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

# Every queue the main compose routes: a job (or a beat tick) sent to a queue the harness does not
# consume would stay `queued` forever, and tests/live/test_queue_delivery.py proves each one.
NEEDED = TaskQueues.ALL


def test_the_harness_has_a_consumer_for_every_queue_the_live_checks_use() -> None:
    services = yaml.safe_load(HARNESS.read_text(encoding='utf-8'))['services']
    found = consumers(services)
    assert [q for q in NEEDED if q not in found] == []


def test_the_harness_beat_is_restarted_after_it_exits() -> None:
    """Beat exits when it loses its database lock connection (a database restart), so that the
    lock is released; the supervisor starts it again. Without the policy no reaper tick ever runs
    after a database restart and a job whose lease expired stays `running`."""
    services = yaml.safe_load(HARNESS.read_text(encoding='utf-8'))['services']
    assert services['jobs-beat'].get('restart') == 'unless-stopped'
