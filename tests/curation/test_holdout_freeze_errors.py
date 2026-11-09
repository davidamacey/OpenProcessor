"""The freeze service fails closed and types its refusals before it touches the database."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services.curation import holdout_freeze as svc


_NO_SESSION: Any = None


@pytest.mark.asyncio
async def test_an_existing_holdout_refuses_a_rerun_without_force() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 3})
    with pytest.raises(svc.HoldoutExistsError):
        await svc.freeze_test_holdout(fake, _NO_SESSION, percent=10, force=False)
    fake.search.assert_not_called()


@pytest.mark.asyncio
async def test_an_unanswerable_guard_refuses_the_freeze() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(side_effect=RuntimeError('connection refused'))
    with pytest.raises(svc.HoldoutCheckError):
        await svc.freeze_test_holdout(fake, _NO_SESSION, percent=10, force=False)
    fake.search.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_cohort_read_is_a_search_error_and_force_skips_the_guard() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(side_effect=AssertionError('force must not count'))
    fake.search = AsyncMock(side_effect=RuntimeError('connection refused'))
    with pytest.raises(svc.HoldoutSearchError):
        await svc.freeze_test_holdout(fake, _NO_SESSION, percent=10, force=True)


@pytest.mark.asyncio
async def test_an_empty_cohort_freezes_nothing() -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'aggregations': {'strata': {'buckets': []}}})
    with pytest.raises(svc.EmptyHoldoutCohortError, match='refusing to freeze nothing'):
        await svc.freeze_test_holdout(fake, _NO_SESSION, percent=10, force=True)
