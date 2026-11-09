"""Plugin: the job-test fixtures, with the fake broker and wake-ups active in every db test.

A plugin named in ``pytest_plugins`` is registered for the whole session, so the autouse
fixture applies the fake only to ``db`` tests (the only ones that start jobs); the others
need no database.
"""

from __future__ import annotations

import pytest


pytest_plugins = ['curation.jobs_a.optin_queue']


@pytest.fixture(autouse=True)
def _fake_queue(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker('db') is not None:
        request.getfixturevalue('fake_queue')
