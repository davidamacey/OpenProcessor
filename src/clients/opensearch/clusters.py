"""Embedding extraction and cluster album queries."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from opensearchpy.helpers import async_bulk

from src.clients.opensearch.names import IndexName


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = logging.getLogger(__name__)


class ClustersMixin:
    """Embedding extraction and cluster album queries."""

    client: AsyncOpenSearch
    embedding_dim: int

    # =========================================================================
    # Embedding Extraction (for Cluster Training/Rebalancing)
    # =========================================================================

    async def get_all_embeddings(
        self,
        index_name: IndexName,
        batch_size: int = 1000,
        max_docs: int | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract all embeddings from an index for cluster training.

        Uses scroll API for efficient large-scale extraction.

        Args:
            index_name: Which index to extract from
            batch_size: Documents per scroll batch
            max_docs: Maximum documents to extract (None = all)

        Returns:
            Tuple of (embeddings array [N, 512], document IDs list)
        """
        embedding_field = 'global_embedding' if index_name == IndexName.GLOBAL else 'embedding'
        id_field = 'image_id' if index_name == IndexName.GLOBAL else 'detection_id'

        embeddings = []
        doc_ids = []

        try:
            # Initialize scroll
            response = await self.client.search(
                index=index_name.value,
                body={
                    'size': batch_size,
                    '_source': [embedding_field, id_field],
                    'query': {'match_all': {}},
                },
                scroll='5m',
            )

            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']

            while hits:
                for hit in hits:
                    source = hit['_source']
                    if embedding_field in source:
                        embeddings.append(source[embedding_field])
                        doc_ids.append(source.get(id_field, hit['_id']))

                        if max_docs and len(embeddings) >= max_docs:
                            break

                if max_docs and len(embeddings) >= max_docs:
                    break

                # Get next batch
                response = await self.client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

            # Clear scroll
            await self.client.clear_scroll(scroll_id=scroll_id)

            logger.info(f'Extracted {len(embeddings)} embeddings from {index_name.value}')
            return np.array(embeddings, dtype=np.float32), doc_ids

        except Exception as e:
            logger.error(f'Failed to extract embeddings from {index_name.value}: {e}')
            return np.array([]), []

    async def get_unclustered_embeddings(
        self,
        index_name: IndexName,
        batch_size: int = 1000,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Get embeddings that don't have cluster assignments yet.

        Useful for incremental clustering of newly ingested items.

        Returns:
            Tuple of (embeddings array, document IDs)
        """
        embedding_field = 'global_embedding' if index_name == IndexName.GLOBAL else 'embedding'
        id_field = 'image_id' if index_name == IndexName.GLOBAL else 'detection_id'

        embeddings = []
        doc_ids = []

        try:
            response = await self.client.search(
                index=index_name.value,
                body={
                    'size': batch_size,
                    '_source': [embedding_field, id_field],
                    'query': {'bool': {'must_not': {'exists': {'field': 'cluster_id'}}}},
                },
                scroll='5m',
            )

            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']

            while hits:
                for hit in hits:
                    source = hit['_source']
                    if embedding_field in source:
                        embeddings.append(source[embedding_field])
                        doc_ids.append(source.get(id_field, hit['_id']))

                response = await self.client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

            await self.client.clear_scroll(scroll_id=scroll_id)

            logger.info(f'Found {len(embeddings)} unclustered items in {index_name.value}')
            return np.array(embeddings, dtype=np.float32), doc_ids

        except Exception as e:
            logger.error(f'Failed to get unclustered embeddings from {index_name.value}: {e}')
            return np.array([]), []

    async def update_cluster_assignments(
        self,
        index_name: IndexName,
        doc_ids: list[str],
        cluster_ids: list[int],
        cluster_distances: list[float],
    ) -> int:
        """
        Bulk update cluster assignments for documents.

        Args:
            index_name: Which index to update
            doc_ids: List of document IDs
            cluster_ids: Corresponding cluster IDs
            cluster_distances: Corresponding distances to centroids

        Returns:
            Number of successfully updated documents
        """
        if not doc_ids:
            return 0

        timestamp = datetime.now(UTC).isoformat()
        actions = []

        for doc_id, cluster_id, distance in zip(
            doc_ids, cluster_ids, cluster_distances, strict=True
        ):
            actions.append(
                {
                    '_op_type': 'update',
                    '_index': index_name.value,
                    '_id': doc_id,
                    'doc': {
                        'cluster_id': cluster_id,
                        'cluster_distance': distance,
                        'clustered_at': timestamp,
                    },
                }
            )

        try:
            success, errors = await async_bulk(self.client, actions, raise_on_error=False)
            if errors:
                logger.warning(f'Some cluster updates failed: {len(errors)} errors')
            logger.info(f'Updated {success} cluster assignments in {index_name.value}')
            return success
        except Exception as e:
            logger.error(f'Bulk cluster update failed: {e}')
            return 0

    # =========================================================================
    # Cluster Album Queries
    # =========================================================================

    async def get_cluster_members(
        self,
        index_name: IndexName,
        cluster_id: int,
        page: int = 0,
        size: int = 50,
        sort_by_distance: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get all members of a specific cluster (like a Google Photos album).

        Args:
            index_name: Which index to query
            cluster_id: Cluster ID to retrieve
            page: Page number (0-indexed)
            size: Page size
            sort_by_distance: If True, sort by distance to centroid (most representative first)

        Returns:
            List of documents in the cluster
        """
        embedding_field = 'global_embedding' if index_name == IndexName.GLOBAL else 'embedding'

        try:
            query = {
                'query': {'term': {'cluster_id': cluster_id}},
                'from': page * size,
                'size': size,
                '_source': {'excludes': [embedding_field]},  # Don't return large embeddings
            }

            if sort_by_distance:
                query['sort'] = [
                    {
                        'cluster_distance': {
                            'order': 'asc',
                            'missing': '_last',
                            'unmapped_type': 'double',
                        }
                    }
                ]

            response = await self.client.search(index=index_name.value, body=query)

            results = []
            for hit in response['hits']['hits']:
                result = {'score': hit.get('_score'), **hit['_source']}
                results.append(result)

            return results

        except Exception as e:
            logger.error(f'Failed to get cluster members: {e}')
            return []

    async def get_cluster_stats(
        self,
        index_name: IndexName,
    ) -> list[dict[str, Any]]:
        """
        Get statistics about clusters in an index.

        Returns:
            List of cluster stats with id, count, avg_distance
        """
        try:
            response = await self.client.search(
                index=index_name.value,
                body={
                    'size': 0,
                    'aggs': {
                        'clusters': {
                            'terms': {
                                'field': 'cluster_id',
                                'size': 10000,  # Get all clusters
                            },
                            'aggs': {
                                'avg_distance': {'avg': {'field': 'cluster_distance'}},
                                'min_distance': {'min': {'field': 'cluster_distance'}},
                            },
                        }
                    },
                },
            )

            return [
                {
                    'cluster_id': bucket['key'],
                    'count': bucket['doc_count'],
                    'avg_distance': bucket['avg_distance']['value'],
                    'min_distance': bucket['min_distance']['value'],
                }
                for bucket in response['aggregations']['clusters']['buckets']
            ]

        except Exception as e:
            logger.error(f'Failed to get cluster stats: {e}')
            return []
