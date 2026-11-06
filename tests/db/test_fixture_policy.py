"""The database fixture fails, never skips, when no database can be reached."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fixtures.postgres import DatabaseUnavailable, pg_server


pytestmark = pytest.mark.db


def _standalone():
    """A request whose config carries no xdist handover, as in a single process."""
    return SimpleNamespace(config=SimpleNamespace())


def test_a_closed_port_raises_database_unavailable_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_TEST_DATABASE_URL', 'postgresql://op_super:x@127.0.0.1:9/postgres')
    # Call the fixture function itself: the real one is session scoped and already resolved.
    with pytest.raises(DatabaseUnavailable, match='unreachable'):
        pg_server.__wrapped__(_standalone())


def test_a_url_without_credentials_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_TEST_DATABASE_URL', 'postgresql://127.0.0.1:5432/postgres')
    with pytest.raises(DatabaseUnavailable, match='OP_TEST_DATABASE_URL'):
        pg_server.__wrapped__(_standalone())
