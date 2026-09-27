"""SSE stats fan-out.

``_cached_stats_payload`` is a module-level, TTL-cached wrapper around
``stats_dataset`` shared by every open SSE connection. It must:

- issue exactly one OpenSearch query for concurrent calls within the
  TTL window (no per-connection stampede), and
- issue a new query once the TTL has expired.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from src.config.project_context import bind_project
from src.routers.curation import pipeline_events as pe
from src.services.projects.registry import default_project_record


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_stats_cache() -> Iterator[None]:
    """Each test starts with a cold cache — module-level state persists
    across tests otherwise."""
    pe._stats_cache_payload = {}
    pe._stats_cache_expires_at = {}
    yield
    pe._stats_cache_payload = {}
    pe._stats_cache_expires_at = {}


@pytest.fixture(autouse=True)
def _bind_default_project() -> Iterator[None]:
    """_cached_stats_payload keys its cache by the bound project's slug."""
    with bind_project(default_project_record()):
        yield


@pytest.mark.asyncio
async def test_concurrent_calls_within_ttl_hit_opensearch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def _fake_stats_dataset(_opensearch: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)  # simulate a real round-trip
        return {'total_crops': calls}

    monkeypatch.setattr('src.routers.curation.stats.stats_dataset', _fake_stats_dataset)

    os_client = AsyncMock()
    results = await asyncio.gather(*(pe._cached_stats_payload(os_client) for _ in range(8)))

    assert calls == 1
    assert all(r == {'total_crops': 1} for r in results)


@pytest.mark.asyncio
async def test_new_query_issued_after_ttl_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def _fake_stats_dataset(_opensearch: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {'total_crops': calls}

    monkeypatch.setattr('src.routers.curation.stats.stats_dataset', _fake_stats_dataset)
    monkeypatch.setattr(pe, '_STATS_CACHE_TTL_SECONDS', 0.0)  # expire immediately

    os_client = AsyncMock()
    first = await pe._cached_stats_payload(os_client)
    second = await pe._cached_stats_payload(os_client)

    assert calls == 2
    assert first == {'total_crops': 1}
    assert second == {'total_crops': 2}


@pytest.mark.asyncio
async def test_within_ttl_second_call_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def _fake_stats_dataset(_opensearch: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {'total_crops': calls}

    monkeypatch.setattr('src.routers.curation.stats.stats_dataset', _fake_stats_dataset)

    os_client = AsyncMock()
    first = await pe._cached_stats_payload(os_client)
    second = await pe._cached_stats_payload(os_client)

    assert calls == 1
    assert first == second == {'total_crops': 1}


@pytest.mark.asyncio
async def test_opensearch_error_is_cached_as_error_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(_opensearch: Any) -> dict[str, Any]:
        raise RuntimeError('opensearch down')

    monkeypatch.setattr('src.routers.curation.stats.stats_dataset', _boom)

    os_client = AsyncMock()
    payload = await pe._cached_stats_payload(os_client)
    assert 'error' in payload
    assert 'opensearch down' in payload['error']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
