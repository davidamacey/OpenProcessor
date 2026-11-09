"""The database a Harness test runs on, for tests outside this directory (whose conftest does
the same for its own tests)."""

from __future__ import annotations

import pytest
from _records_world import world


@pytest.fixture
def import_database(request: pytest.FixtureRequest) -> None:
    """The migrated database with the bound project's row and the harness's handle on it."""
    from curation.dataset_import import harness

    pg = request.getfixturevalue('pg_env')
    request.getfixturevalue('world')
    harness.PG = pg


__all__ = ['import_database', 'world']
