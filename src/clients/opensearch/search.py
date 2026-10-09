"""Category k-NN search and index statistics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.clients.opensearch.names import IndexMappingError, IndexName


if TYPE_CHECKING:
    import numpy as np
    from opensearchpy import AsyncOpenSearch


logger = logging.getLogger(__name__)


class SearchMixin:
    """Category k-NN search and index statistics."""

    client: AsyncOpenSearch
    embedding_dim: int

    # =========================================================================
    # Category-Specific Search Methods
    # =========================================================================

    async def search_global(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar images by whole-image embedding.

        Args:
            query_embedding: 512-dim L2-normalized query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            cluster_ids: Optional list of cluster IDs to narrow search (from ClusteringService)

        Returns:
            List of image results with scores
        """
        return await self._search_index(
            index_name=IndexName.GLOBAL.value,
            embedding_field='global_embedding',
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            cluster_ids=cluster_ids,
        )

    async def search_vehicles(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vehicles by embedding.

        Args:
            query_embedding: 512-dim L2-normalized query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            class_filter: Filter by specific vehicle classes (e.g., [2] for cars only)
            cluster_ids: Optional list of cluster IDs to narrow search

        Returns:
            List of vehicle detection results with scores
        """
        return await self._search_index(
            index_name=IndexName.VEHICLES.value,
            embedding_field='embedding',
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            class_filter=class_filter,
            cluster_ids=cluster_ids,
        )

    async def search_people(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar people by appearance embedding.

        Args:
            query_embedding: 512-dim L2-normalized query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            cluster_ids: Optional list of cluster IDs to narrow search

        Returns:
            List of person detection results with scores
        """
        return await self._search_index(
            index_name=IndexName.PEOPLE.value,
            embedding_field='embedding',
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            cluster_ids=cluster_ids,
        )

    async def search_faces(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = 0.7,  # Higher threshold for identity matching
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for same person by face embedding (identity matching).

        Args:
            query_embedding: 512-dim ArcFace embedding
            top_k: Number of results to return
            min_score: Minimum similarity score (default 0.7 for identity)
            cluster_ids: Optional list of cluster IDs to narrow search

        Returns:
            List of face results with identity info
        """
        return await self._search_index(
            index_name=IndexName.FACES.value,
            embedding_field='embedding',
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            cluster_ids=cluster_ids,
        )

    async def get_faces_by_person_id(
        self,
        person_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get all faces belonging to a person identity."""
        query = {
            'query': {'term': {'person_id': person_id}},
            'size': limit,
            'sort': [{'confidence': {'order': 'desc'}}],
        }
        response = await self.client.search(
            index=IndexName.FACES.value,
            body=query,
        )
        return [hit['_source'] for hit in response['hits']['hits']]

    async def _search_index(
        self,
        index_name: str,
        embedding_field: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Internal helper for k-NN search on any index.

        Args:
            index_name: OpenSearch index name
            embedding_field: Field name containing the embedding vector
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            class_filter: Filter by class IDs (for vehicle searches)
            cluster_ids: Filter by cluster IDs (for cluster-optimized search)

        When cluster_ids is provided, OpenSearch only searches documents
        in those clusters, typically providing 10-100x speedup for large indexes.
        """
        try:
            query = {
                'size': top_k,
                'query': {
                    'knn': {
                        embedding_field: {
                            'vector': query_embedding.tolist(),
                            'k': top_k,
                        }
                    }
                },
            }

            # Build filter conditions
            filters = []
            if class_filter:
                filters.append({'terms': {'class_id': class_filter}})
            if cluster_ids:
                filters.append({'terms': {'cluster_id': cluster_ids}})

            # Apply filters if any
            if filters:
                query['query'] = {
                    'bool': {
                        'must': [query['query']],
                        'filter': filters,
                    }
                }

            if min_score is not None:
                query['min_score'] = min_score

            response = await self.client.search(index=index_name, body=query)

            results = []
            for hit in response['hits']['hits']:
                result = {
                    'score': hit['_score'],
                    **hit['_source'],
                }
                # Remove embedding from results (too large)
                result.pop(embedding_field, None)
                result.pop('embedding', None)
                result.pop('global_embedding', None)
                results.append(result)

            return results

        except Exception as e:
            message = str(e)
            if 'knn_vector' in message and 'is not' in message:
                # F-25: don't swallow this one -- an unmapped/misconfigured
                # index is not the same thing as "no similar images", and
                # the caller (a router that turns an unhandled exception
                # into a 500) needs to see it as an error, not a success
                # with an empty results list.
                logger.error(f'{index_name}.{embedding_field} is not knn_vector-mapped: {e}')
                raise IndexMappingError(
                    f"'{index_name}' exists but '{embedding_field}' is not a knn_vector "
                    f'field -- it was likely auto-created by a write before the index was '
                    f'properly initialized. Delete it and restart the API (startup now '
                    f'calls create_all_indexes on every boot), or call '
                    f'OpenSearchClient.create_all_indexes(force_recreate=True) directly.'
                ) from e
            logger.error(f'Search failed on {index_name}: {e}')
            return []

    async def get_all_index_stats(self) -> dict[str, Any]:
        """``{index: {doc_count, size_bytes, exists}}`` for every core index
        (the shape ``GET /query/stats`` serves).

        The count comes from ``_count`` -- a search, so an idle shard is
        refreshed first and a just-ingested document is counted -- where
        ``_stats`` reports only what the last refresh made searchable.
        """
        stats: dict[str, Any] = {}
        for index_name in IndexName:
            name = index_name.value
            try:
                if not await self.client.indices.exists(index=name):
                    stats[name] = {'exists': False, 'doc_count': 0, 'size_bytes': 0}
                    continue
                count = await self.client.count(index=name)
                response = await self.client.indices.stats(index=name)
                stats[name] = {
                    'exists': True,
                    'doc_count': int(count['count']),
                    'size_bytes': int(response['_all']['primaries']['store']['size_in_bytes']),
                }
            except Exception as e:
                logger.error(f'Index stats failed for {name}: {e}')
                stats[name] = {'exists': False, 'doc_count': 0, 'size_bytes': 0, 'error': str(e)}
        return stats
