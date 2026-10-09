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

from src.clients.opensearch.client import OpenSearchClient
from src.clients.opensearch.names import IndexMappingError, IndexName


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


class TestIndexCreation:
    """Found live: OpenSearch 3.6 rejected the OCR index (ngram diff 2 > the
    default max_ngram_diff of 1), so it never existed; and the N API workers
    racing to create the same index logged the loser's
    ``resource_already_exists_exception`` as an error."""

    @pytest.mark.asyncio
    async def test_ocr_index_allows_its_ngram_filter_width(self, client: OpenSearchClient) -> None:
        client.client.indices.exists = AsyncMock(return_value=False)
        client.client.indices.create = AsyncMock()

        assert await client.create_ocr_index() is True

        call = client.client.indices.create.await_args
        assert call is not None
        body = call.kwargs['body']
        ngram = body['settings']['analysis']['filter']['trigram_filter']
        allowed = body['settings']['index']['max_ngram_diff']
        assert ngram['max_gram'] - ngram['min_gram'] <= allowed

    @pytest.mark.asyncio
    async def test_losing_the_create_race_is_success(
        self, client: OpenSearchClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        from opensearchpy.exceptions import RequestError

        client.client.indices.exists = AsyncMock(return_value=False)
        client.client.indices.create = AsyncMock(
            side_effect=RequestError(400, 'resource_already_exists_exception', {})
        )

        with caplog.at_level('ERROR'):
            assert await client.create_global_index() is True
        assert not [r for r in caplog.records if r.levelname == 'ERROR']

    @pytest.mark.asyncio
    async def test_other_create_errors_still_fail(self, client: OpenSearchClient) -> None:
        from opensearchpy.exceptions import RequestError

        client.client.indices.exists = AsyncMock(return_value=False)
        client.client.indices.create = AsyncMock(
            side_effect=RequestError(400, 'illegal_argument_exception', {})
        )

        assert await client.create_global_index() is False


class TestIndexStats:
    """Found live: ``GET /query/stats`` reported 0 documents for every index
    (the client returned ``total_documents``; the router reads
    ``doc_count``) and never listed the OCR index."""

    @pytest.mark.asyncio
    async def test_every_core_index_reports_the_shape_the_router_reads(
        self, client: OpenSearchClient
    ) -> None:
        client.client.indices.exists = AsyncMock(return_value=True)
        client.client.count = AsyncMock(return_value={'count': 7})
        client.client.indices.stats = AsyncMock(
            return_value={'_all': {'primaries': {'store': {'size_in_bytes': 2048}}}}
        )

        stats = await client.get_all_index_stats()

        assert set(stats) == {index.value for index in IndexName}
        assert IndexName.OCR.value in stats
        for entry in stats.values():
            assert entry == {'exists': True, 'doc_count': 7, 'size_bytes': 2048}

    @pytest.mark.asyncio
    async def test_a_missing_index_reports_zero(self, client: OpenSearchClient) -> None:
        client.client.indices.exists = AsyncMock(return_value=False)

        stats = await client.get_all_index_stats()

        assert stats[IndexName.GLOBAL.value] == {'exists': False, 'doc_count': 0, 'size_bytes': 0}


class TestOcrSearch:
    """Found live: ``POST /ocr/search`` queried ``full_text``/``texts`` -- fields
    the OCR index does not have -- and ``/search/ocr`` read keys the client
    never returned, so OCR search matched nothing useful."""

    @pytest.mark.asyncio
    async def test_queries_the_text_fields_the_index_has(self, client: OpenSearchClient) -> None:
        client.client.search = AsyncMock(
            return_value={
                'hits': {
                    'hits': [
                        {
                            '_score': 1.5,
                            '_source': {
                                'image_id': 'img-1',
                                'image_path': 'a.jpg',
                                'text': 'CAUTION',
                                'box_normalized': [0.1, 0.2, 0.3, 0.4],
                            },
                        }
                    ]
                },
                'aggregations': {'images': {'value': 7}},
            }
        )

        results, total = await client.search_ocr_page('caution', offset=20, size=10)

        call = client.client.search.await_args
        assert call is not None
        body = call.kwargs['body']
        assert body['from'] == 20
        assert body['size'] == 10
        assert 'match' in body['query']['bool']['should'][0]
        assert 'text' in body['query']['bool']['should'][0]['match']
        assert total == 7
        assert results[0]['text'] == 'CAUTION'
        assert results[0]['image_id'] == 'img-1'
        assert results[0]['box_normalized'] == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_exact_matches_the_whole_line_only(self, client: OpenSearchClient) -> None:
        client.client.search = AsyncMock(return_value={'hits': {'hits': []}})

        await client.search_ocr_page('STOP', exact=True)

        call = client.client.search.await_args
        assert call is not None
        body = call.kwargs['body']
        assert body['query']['bool']['should'] == [{'term': {'text_raw': {'value': 'STOP'}}}]
