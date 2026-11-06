"""The database fixture fails, never skips, when no database can be reached."""

from __future__ import annotations

import pytest
from fixtures.postgres import DatabaseUnavailable, pg_server


pytestmark = pytest.mark.db


def test_a_closed_port_raises_database_unavailable_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv('OP_TEST_DATABASE_URL', 'postgresql://op_super:x@127.0.0.1:9/postgres')
    # Call the fixture function itself: the real one is session scoped and already resolved.
    with pytest.raises(DatabaseUnavailable, match='unreachable'):
        pg_server.__wrapped__(request)


def test_a_url_without_credentials_is_refused(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv('OP_TEST_DATABASE_URL', 'postgresql://127.0.0.1:5432/postgres')
    with pytest.raises(DatabaseUnavailable, match='OP_TEST_DATABASE_URL'):
        pg_server.__wrapped__(request)
