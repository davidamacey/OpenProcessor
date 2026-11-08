"""Plugin: the job-test fixtures, with the fake broker and wake-ups active in every test."""

from __future__ import annotations

import pytest

from curation.jobs_a.optin_queue import fake_queue, job_env, kinds, producer, queue_hooks, wakes  # noqa: F401


@pytest.fixture(autouse=True)
def _fake_queue(fake_queue) -> None:  # noqa: F811 - set up by fixture name
    """Every test of a module that loads this plugin starts jobs against the fake."""
