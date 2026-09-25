"""F-25 (fresh-start E2E findings 2026-09-25): core visual-search indexes
must not fail open.

Nothing called ``OpenSearchClient.create_all_indexes()`` at startup, so on
a fresh install the first ``POST /ingest`` auto-created each core
``visual_search_*`` index with OpenSearch's dynamic mapping instead
(``"global_embedding": {"type": "float"}``, not ``knn_vector``). Every
k-NN search against that index then failed with an OpenSearch error
("Field 'global_embedding' is not knn_vector type"), which
``_search_index`` swallowed into an empty list -- indistinguishable from
"no similar images", returned as ``status: success`` by the router.

Two things are tested here without a real OpenSearch:

1. ``_search_index`` now raises :class:`IndexMappingError` instead of
   swallowing that specific error class, while still swallowing other,
   genuinely transient search failures the old way (unchanged behavior).
2. ``OpenSearchClient.create_all_indexes()`` is idempotent (a no-op
   against an index that already exists), so calling it on every startup
   -- the actual fix in ``src/main.py`` -- is safe.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.clients.opensearch import IndexMappingError, IndexName, OpenSearchClient


@pytest.fixture
def client() -> OpenSearchClient:
    os_client = OpenSearchClient(hosts=['http://localhost:9200'])
    os_client.client = MagicMock()
    os_client.client.indices = MagicMock()
    return os_client


class TestSearchIndexSurfacesMappingErrors:
    @pytest.mark.asyncio
    async def test_knn_vector_mapping_error_is_raised_not_swallowed(
        self, client: OpenSearchClient
    ) -> None:
        client.client.search = AsyncMock(
            side_effect=RuntimeError(
                "Field 'global_embedding' is not knn_vector type. "
                'ScriptScoreQueryBuilder does not support ...'
            )
        )
        with pytest.raises(IndexMappingError, match='global_embedding'):
            await client.search_global(np.zeros(512, dtype=np.float32), top_k=5)

    @pytest.mark.asyncio
    async def test_an_unrelated_search_failure_still_degrades_to_empty(
        self, client: OpenSearchClient
    ) -> None:
        """Genuinely transient errors (timeouts, connection refused, ...)
        keep the pre-existing fail-soft behavior -- this fix is scoped to
        the specific mapping-type failure, not a blanket 'raise on
        anything' change."""
        client.client.search = AsyncMock(side_effect=ConnectionError('connection refused'))
        results = await client.search_global(np.zeros(512, dtype=np.float32), top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_a_healthy_index_returns_real_results(self, client: OpenSearchClient) -> None:
        client.client.search = AsyncMock(
            return_value={
                'hits': {
                    'hits': [
                        {'_score': 0.98, '_source': {'image_id': 'abc', 'global_embedding': []}}
                    ]
                }
            }
        )
        results = await client.search_global(np.zeros(512, dtype=np.float32), top_k=5)
        assert results == [{'score': 0.98, 'image_id': 'abc'}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'search_method',
        ['search_global', 'search_vehicles', 'search_people', 'search_faces'],
    )
    async def test_every_core_knn_search_method_surfaces_the_mapping_error(
        self, client: OpenSearchClient, search_method: str
    ) -> None:
        """All four core indexes route through the same _search_index
        helper, so the fix covers all of them, not just visual_search_global."""
        client.client.search = AsyncMock(
            side_effect=RuntimeError("Field 'embedding' is not knn_vector type")
        )
        method = getattr(client, search_method)
        with pytest.raises(IndexMappingError):
            await method(np.zeros(512, dtype=np.float32), top_k=5)


class TestCreateAllIndexesIsIdempotent:
    @pytest.mark.asyncio
    async def test_an_existing_index_is_left_alone_by_default(
        self, client: OpenSearchClient
    ) -> None:
        client.client.indices.exists = AsyncMock(return_value=True)
        client.client.indices.create = AsyncMock()
        results = await client.create_all_indexes(force_recreate=False)
        assert all(results.values()), results
        client.client.indices.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_missing_index_is_created(self, client: OpenSearchClient) -> None:
        client.client.indices.exists = AsyncMock(return_value=False)
        client.client.indices.create = AsyncMock(return_value={'acknowledged': True})
        results = await client.create_all_indexes(force_recreate=False)
        assert all(results.values()), results
        assert client.client.indices.create.call_count == len(IndexName)
