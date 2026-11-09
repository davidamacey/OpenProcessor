"""Plugin: the fake broker and wake-ups as an opt-in fixture, ``usefixtures('fake_queue')``.

A module that starts jobs names ``fake_queue`` in its ``pytestmark`` (or on the db tests of a
mixed module). It is never autouse: a plugin named in a test module's ``pytest_plugins`` is
registered for the whole session, so an autouse fixture there would reach every ``db`` test of
the run, including those that assert on an unbound engine or on their own producer.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_queue(producer, wakes, job_env) -> None:
    """Starts send to a recording fake and nothing reaches a broker."""
