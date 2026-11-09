"""Plugin: the fake broker and wake-ups as an opt-in fixture, ``usefixtures('fake_queue')``.

For a db test in a module that also holds tests with no database; a module that is all db
tests loads ``autoqueue`` instead, which applies it to every test.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_queue(producer, wakes, job_env) -> None:
    """Starts send to a recording fake and nothing reaches a broker."""
