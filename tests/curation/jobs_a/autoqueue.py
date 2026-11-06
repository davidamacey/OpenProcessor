"""Plugin: the job-test fixtures, with the fake broker and wake-ups active in every test."""

from __future__ import annotations

import pytest
from fixtures.job_fixtures import job_env, kinds, producer, queue_hooks, wakes  # noqa: F401


@pytest.fixture(autouse=True)
def _fake_queue(producer, wakes, job_env) -> None:  # noqa: F811 - set up by fixture name
    """Starts send to a recording fake and nothing reaches a broker."""
