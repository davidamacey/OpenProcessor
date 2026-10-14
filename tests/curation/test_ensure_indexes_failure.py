"""A project whose index migrations did not all succeed is not marked bootstrapped,
so the next request retries them instead of serving on an unmigrated index."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.config import get_curation_config
from src.services.curation import index_bootstrap


class _OpenSearch:
    async def count(self, **_: Any) -> dict[str, int]:
        return {'count': 1}


@pytest.fixture(autouse=True)
def _bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', set())
    monkeypatch.setattr(index_bootstrap, 'create_curation_indexes', AsyncMock())
    ok = AsyncMock(return_value={'acknowledged': True})
    for name in (
        n for n in vars(index_bootstrap) if n.startswith('ensure_') and n != 'ensure_indexes'
    ):
        monkeypatch.setattr(index_bootstrap, name, ok)


@pytest.mark.asyncio
async def test_all_migrations_ok_marks_the_project_bootstrapped() -> None:
    await index_bootstrap.ensure_indexes(_OpenSearch())
    assert get_curation_config().project_slug in index_bootstrap._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_a_raising_migration_leaves_the_project_unbootstrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        index_bootstrap,
        'ensure_items_text_fields',
        AsyncMock(side_effect=RuntimeError('mapping down')),
    )
    await index_bootstrap.ensure_indexes(_OpenSearch())
    assert get_curation_config().project_slug not in index_bootstrap._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_an_unacknowledged_migration_leaves_the_project_unbootstrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The ensure_* helpers report a failed put_mapping as a result, not an exception.
    failed = AsyncMock(return_value={'acknowledged': False, 'error': 'conflict'})
    monkeypatch.setattr(index_bootstrap, 'ensure_items_score_fields', failed)
    await index_bootstrap.ensure_indexes(_OpenSearch())
    assert get_curation_config().project_slug not in index_bootstrap._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_the_next_request_retries_and_then_bootstraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flaky = AsyncMock(side_effect=[{'acknowledged': False, 'error': 'x'}, {'acknowledged': True}])
    monkeypatch.setattr(index_bootstrap, 'ensure_items_score_fields', flaky)

    await index_bootstrap.ensure_indexes(_OpenSearch())
    assert get_curation_config().project_slug not in index_bootstrap._INDEXES_BOOTSTRAPPED
    await index_bootstrap.ensure_indexes(_OpenSearch())

    assert flaky.await_count == 2
    assert get_curation_config().project_slug in index_bootstrap._INDEXES_BOOTSTRAPPED
