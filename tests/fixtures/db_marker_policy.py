"""Collection policy of the ``db`` tier: a test that needs the database carries the marker.

A test under ``tests/db/`` or one that uses the ``pg`` fixture (directly or through
``pg_env`` / ``two_sessions``) without ``@pytest.mark.db`` would run in the default selection
and fail there, or worse, be skipped. Collection stops with a usage error naming it instead.
Hook reference: https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_collection_modifyitems
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import PurePath
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import Sequence

DB_FIXTURES = frozenset({'pg', 'pg_env', 'pg_server', 'two_sessions'})


def _needs_database(item: pytest.Item) -> bool:
    parts = PurePath(str(item.path)).parts
    in_db_dir = any(a == 'tests' and b == 'db' for a, b in pairwise(parts))
    return in_db_dir or bool(DB_FIXTURES & set(getattr(item, 'fixturenames', ())))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: Sequence[pytest.Item]) -> None:
    unmarked = [
        item.nodeid
        for item in items
        if _needs_database(item) and item.get_closest_marker('db') is None
    ]
    if unmarked:
        listing = '\n  '.join(unmarked)
        raise pytest.UsageError(
            f'these tests need the database but lack @pytest.mark.db '
            f'(add pytestmark = pytest.mark.db):\n  {listing}'
        )
