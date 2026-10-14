"""A project whose index migrations did not all succeed is not marked bootstrapped,
so the next request retries them instead of serving on an unmigrated index."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.routers.curation import _common


class _OpenSearch:
    async def count(self, **_: Any) -> dict[str, int]:
        return {'count': 1}


@pytest.fixture(autouse=True)
def _bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', set())
    monkeypatch.setattr(_common, 'create_curation_indexes', AsyncMock())
    ok = AsyncMock(return_value={'acknowledged': True})
    for name in (n for n in vars(_common) if n.startswith('ensure_')):
        monkeypatch.setattr(_common, name, ok)


@pytest.mark.asyncio
async def test_all_migrations_ok_marks_the_project_bootstrapped() -> None:
    await _common._ensure_indexes(_OpenSearch())
    assert _common.config.project_slug in _common._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_a_raising_migration_leaves_the_project_unbootstrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _common, 'ensure_items_text_fields', AsyncMock(side_effect=RuntimeError('mapping down'))
    )
    await _common._ensure_indexes(_OpenSearch())
    assert _common.config.project_slug not in _common._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_an_unacknowledged_migration_leaves_the_project_unbootstrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The ensure_* helpers report a failed put_mapping as a result, not an exception.
    failed = AsyncMock(return_value={'acknowledged': False, 'error': 'conflict'})
    monkeypatch.setattr(_common, 'ensure_items_score_fields', failed)
    await _common._ensure_indexes(_OpenSearch())
    assert _common.config.project_slug not in _common._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_the_next_request_retries_and_then_bootstraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flaky = AsyncMock(side_effect=[{'acknowledged': False, 'error': 'x'}, {'acknowledged': True}])
    monkeypatch.setattr(_common, 'ensure_items_score_fields', flaky)

    await _common._ensure_indexes(_OpenSearch())
    assert _common.config.project_slug not in _common._INDEXES_BOOTSTRAPPED
    await _common._ensure_indexes(_OpenSearch())

    assert flaky.await_count == 2
    assert _common.config.project_slug in _common._INDEXES_BOOTSTRAPPED
