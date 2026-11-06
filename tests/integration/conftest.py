"""Fixtures the integration tests of the queued job families use: the fake producer and the
Celery app a worker task runs on."""

from families_b.support import queue_env, worker_app  # noqa: F401  # fixture registration
from fixtures.job_fixtures import FakeProducer, kinds, producer, wakes  # noqa: F401  # fixture registration
