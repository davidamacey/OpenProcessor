"""Fixtures of the family-B tests: the job fixtures, and the real routes over a database."""

from families_b.support import app, client, queue_env, worker_app
from fixtures.job_fixtures import (
    FakeProducer,
    FakeQueueHooks,
    job_env,
    kinds,
    producer,
    queue_hooks,
    wakes,
)


__all__ = [
    'FakeProducer',
    'FakeQueueHooks',
    'app',
    'client',
    'job_env',
    'kinds',
    'producer',
    'queue_env',
    'queue_hooks',
    'wakes',
    'worker_app',
]
