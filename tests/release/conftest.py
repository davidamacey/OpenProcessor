"""Shared fixture of the acceptance-run tests: a factory for fake API servers."""

from __future__ import annotations

import pytest

from tests.release.test_acceptance_run import Fake, _happy_routes


@pytest.fixture
def fake():
    made: list[Fake] = []

    def make(routes=None) -> Fake:
        f = Fake(routes if routes is not None else _happy_routes())
        made.append(f)
        return f

    yield make
    for f in made:
        f.srv.shutdown()
