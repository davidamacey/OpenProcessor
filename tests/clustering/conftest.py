"""Database-tier helpers: the shared seeds live in tests/db, the job fixtures in tests/fixtures."""

import sys
from pathlib import Path

import pytest
from fixtures.job_fixtures import (
    FakeProducer,
    FakeQueueHooks,
    job_env,
    kinds,
    producer,
    queue_hooks,
    wakes,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'db'))


@pytest.fixture(autouse=True)
def _bound_engine_for_db_tests(request: pytest.FixtureRequest) -> None:
    """A ``db`` test calls services directly, as a worker does: bind the worker engine
    (the API-only engine needs ``WEB_CONCURRENCY``, which only the API sets)."""
    if request.node.get_closest_marker('db') is not None:
        request.getfixturevalue('bound_async_engine')


__all__ = ['FakeProducer', 'FakeQueueHooks', 'job_env', 'kinds', 'producer', 'queue_hooks', 'wakes']
