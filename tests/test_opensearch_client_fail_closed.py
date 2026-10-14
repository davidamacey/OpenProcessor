"""The visual-search OpenSearch client raises ``UpstreamError`` on an outage. It used
to return ``[]`` / ``0`` / ``None``, which an outage shares with a real empty answer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.clients.opensearch.client import OpenSearchClient
from src.clients.opensearch.names import IndexName
from src.utils.upstream_errors import UpstreamError


@pytest.fixture
def client() -> OpenSearchClient:
    os_client = OpenSearchClient(hosts=['http://localhost:9200'])
    os_client.client = MagicMock()
    os_client.client.search = AsyncMock(side_effect=ConnectionError('connection refused'))
    os_client.client.count = AsyncMock(side_effect=ConnectionError('connection refused'))
    os_client.client.scroll = AsyncMock(side_effect=ConnectionError('connection refused'))
    os_client.client.bulk = AsyncMock(side_effect=ConnectionError('connection refused'))
    return os_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'method', ['search_global', 'search_vehicles', 'search_people', 'search_faces']
)
async def test_knn_search_outage_raises(client: OpenSearchClient, method: str) -> None:
    with pytest.raises(UpstreamError) as err:
        await getattr(client, method)(np.zeros(512, dtype=np.float32), top_k=5)
    assert err.value.reason == 'opensearch_unavailable'


@pytest.mark.asyncio
async def test_cluster_reads_raise(client: OpenSearchClient) -> None:
    with pytest.raises(UpstreamError):
        await client.get_all_embeddings(IndexName.GLOBAL)
    with pytest.raises(UpstreamError):
        await client.get_unclustered_embeddings(IndexName.GLOBAL)
    with pytest.raises(UpstreamError):
        await client.get_cluster_members(IndexName.GLOBAL, cluster_id=1)
    with pytest.raises(UpstreamError):
        await client.get_cluster_stats(IndexName.GLOBAL)


@pytest.mark.asyncio
async def test_cluster_assignment_write_raises(client: OpenSearchClient, monkeypatch) -> None:
    async def boom(*_a, **_k):
        raise ConnectionError('connection refused')

    monkeypatch.setattr('src.clients.opensearch.clusters.async_bulk', boom)
    with pytest.raises(UpstreamError):
        await client.update_cluster_assignments(IndexName.GLOBAL, ['a'], [1], [0.1])


@pytest.mark.asyncio
async def test_ocr_reads_raise(client: OpenSearchClient) -> None:
    with pytest.raises(UpstreamError):
        await client.search_by_text('plate')
    with pytest.raises(UpstreamError):
        await client.search_ocr('plate')


@pytest.mark.asyncio
async def test_duplicate_checks_raise(client: OpenSearchClient) -> None:
    with pytest.raises(UpstreamError):
        await client.check_duplicate_by_hash('abc')
    with pytest.raises(UpstreamError):
        await client.check_duplicates_by_hash_batch(['abc'])


@pytest.mark.asyncio
async def test_an_empty_answer_is_still_empty(client: OpenSearchClient) -> None:
    client.client.search = AsyncMock(return_value={'hits': {'hits': []}})
    assert await client.search_global(np.zeros(512, dtype=np.float32), top_k=5) == []
    assert await client.get_cluster_members(IndexName.GLOBAL, cluster_id=1) == []


def test_an_outage_reaches_the_caller_as_a_503_with_a_stable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the real app: the OCR text search route answers 503, not an empty 200."""
    from fastapi.testclient import TestClient

    from src.core.dependencies import get_visual_search_service
    from src.main import app

    class _Service:
        class opensearch:  # noqa: N801 - attribute stand-in for the service's client
            @staticmethod
            async def search_ocr(**_: object) -> list[dict[str, object]]:
                raise UpstreamError('opensearch', 'opensearch_unavailable')

    app.dependency_overrides[get_visual_search_service] = _Service
    try:
        response = TestClient(app).post('/search/ocr', params={'text': 'plate'})
    finally:
        app.dependency_overrides.pop(get_visual_search_service, None)

    assert response.status_code == 503
    assert response.json()['reason'] == 'opensearch_unavailable'
    assert response.headers['Retry-After']
