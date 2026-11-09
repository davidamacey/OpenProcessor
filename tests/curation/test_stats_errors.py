"""The stats routes map the services' typed failures: an unbuildable item filter is a 400, a
failed search a 503 (never a zero-filled payload)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytest_plugins = ['_records_world']

BASE = '/curation/projects/default'


def _client(fake: Any) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_class_stats_unbuildable_filter_is_a_400() -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={})
    r = _client(fake).get(BASE + '/stats/classes', params={'min_area': 0.9, 'max_area': 0.1})
    assert r.status_code == 400, r.text
    assert 'min_area' in r.json()['detail']
    fake.search.assert_not_called()


def test_class_stats_failed_search_is_a_503() -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(side_effect=RuntimeError('connection refused'))
    r = _client(fake).get(BASE + '/stats/classes')
    assert r.status_code == 503, r.text
    assert r.json()['detail'] == 'opensearch error'


_NO_SESSION: Any = None


async def _no_repository() -> list[dict[str, Any]]:
    return []


# The dataset service raises its typed errors before it touches the database session.
@pytest.mark.asyncio
async def test_dataset_stats_unbuildable_filter_raises_the_typed_error() -> None:
    from src.services.curation.item_filter import ItemFilter
    from src.services.curation.stats_dataset import build_dataset_stats
    from src.services.curation.stats_dataset_query import InvalidItemFilterError

    fake = AsyncMock()
    with pytest.raises(InvalidItemFilterError, match='min_area'):
        await build_dataset_stats(
            fake,
            ItemFilter(min_area=0.9, max_area=0.1),
            _NO_SESSION,
            repository_index=_no_repository,
        )
    fake.search.assert_not_called()


@pytest.mark.asyncio
async def test_dataset_stats_failed_search_raises_the_typed_error() -> None:
    from src.services.curation.item_filter import ItemFilter
    from src.services.curation.stats_dataset import build_dataset_stats
    from src.services.curation.stats_dataset_query import StatsQueryError

    fake = AsyncMock()
    fake.search = AsyncMock(side_effect=RuntimeError('connection refused'))
    with pytest.raises(StatsQueryError):
        await build_dataset_stats(fake, ItemFilter(), _NO_SESSION, repository_index=_no_repository)
