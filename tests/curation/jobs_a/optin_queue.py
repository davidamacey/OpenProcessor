"""Plugin: the fake broker and wake-ups as an opt-in fixture, ``usefixtures('fake_queue')``.

For a db test in a module that also holds tests with no database; a module that is all db
tests loads ``autoqueue`` instead, which applies it to every test.
"""

from __future__ import annotations

import pytest
from fixtures.job_fixtures import job_env, kinds, producer, queue_hooks, wakes  # noqa: F401


@pytest.fixture
def fake_queue(producer, wakes, job_env) -> None:  # noqa: F811 - set up by fixture name
    """Starts send to a recording fake and nothing reaches a broker."""
