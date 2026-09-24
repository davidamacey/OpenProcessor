"""F-24: kNN warmup startup hook.

``warm_knn_indexes`` (src.routers.curation._common) calls OpenSearch's
kNN warmup endpoint for the items + images indexes. It must never raise
-- a failure (endpoint missing, OS unreachable) is logged and swallowed
so it can safely be fired via ``asyncio.create_task`` without crashing
app startup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.routers.curation._common import (
    CURATION_IMAGES_INDEX,
    CURATION_ITEMS_INDEX,
    warm_knn_indexes,
)


@pytest.mark.asyncio
async def test_warm_knn_indexes_calls_the_warmup_endpoint() -> None:
    os_client = AsyncMock()
    os_client.transport = AsyncMock()
    os_client.transport.perform_request = AsyncMock(return_value={})

    await warm_knn_indexes(os_client)

    os_client.transport.perform_request.assert_awaited_once()
    assert os_client.transport.perform_request.await_args is not None
    method, url = os_client.transport.perform_request.await_args.args
    assert method == 'GET'
    assert url == f'/_plugins/_knn/warmup/{CURATION_ITEMS_INDEX},{CURATION_IMAGES_INDEX}'


@pytest.mark.asyncio
async def test_warm_knn_indexes_swallows_failures() -> None:
    os_client = AsyncMock()
    os_client.transport = AsyncMock()
    os_client.transport.perform_request = AsyncMock(side_effect=RuntimeError('boom'))

    # Must not raise.
    await warm_knn_indexes(os_client)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
