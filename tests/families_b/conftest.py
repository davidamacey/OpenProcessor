"""Fixtures of the family-B tests: the job fixtures, and the real routes over a database."""

import pytest
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


@pytest.fixture(autouse=True)
def _bound_engine_for_db_tests(request: pytest.FixtureRequest) -> None:
    """A ``db`` test calls services directly, as a worker does: bind the worker engine
    (the API-only engine needs ``WEB_CONCURRENCY``, which only the API sets)."""
    if request.node.get_closest_marker('db') is not None:
        request.getfixturevalue('bound_async_engine')
