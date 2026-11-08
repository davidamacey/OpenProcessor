"""The database a Harness test runs on, for tests outside this directory (whose conftest does
the same for its own tests)."""

from __future__ import annotations

import pytest
from _records_world import world


@pytest.fixture
def import_database(request: pytest.FixtureRequest) -> None:
    """The migrated database with the bound project's row, the ledger's ``detail`` column
    (depends on the ND-A merge) and the harness's handle on the database."""
    from families_b.ledger_detail import ensure_detail_column

    from curation.dataset_import import harness

    pg = request.getfixturevalue('pg_env')
    request.getfixturevalue('world')
    ensure_detail_column(pg)
    harness.PG = pg


__all__ = ['import_database', 'world']
