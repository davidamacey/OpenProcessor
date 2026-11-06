"""The seed helpers of tests/db are plain modules, not a package; make them importable here."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections.abc import AsyncIterator

import pytest_asyncio

from src.db.engine import dispose_engines


@pytest_asyncio.fixture
async def repo_env(pg_env) -> AsyncIterator[None]:
    """``pg_env`` plus an awaited engine disposal, so no pool outlives its event loop."""
    yield
    await dispose_engines()
