"""The bake-off tests share the fixtures of ``tests/fixtures/job_fixtures.py``."""

from fixtures.job_fixtures import (
    FakeProducer,
    FakeQueueHooks,
    job_env,
    kinds,
    producer,
    queue_hooks,
    wakes,
)


__all__ = ['FakeProducer', 'FakeQueueHooks', 'job_env', 'kinds', 'producer', 'queue_hooks', 'wakes']
