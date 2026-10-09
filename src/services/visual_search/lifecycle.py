"""Index lifecycle management."""

import logging
from typing import Any

from src.clients.opensearch import OpenSearchClient
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class IndexLifecycleMixin:
    opensearch: OpenSearchClient
    inference: InferenceService

    # =========================================================================
    # Index Management (All Category Indexes)
    # =========================================================================

    async def setup_index(self, force_recreate: bool = False) -> dict[str, bool]:
        """
        Create all visual search indexes (global, vehicles, people, faces).

        Args:
            force_recreate: Delete existing indexes if present

        Returns:
            dict mapping index name to creation success
        """
        return await self.opensearch.create_all_indexes(force_recreate=force_recreate)

    async def delete_index(self) -> dict[str, bool]:
        """Delete all visual search indexes."""
        return await self.opensearch.delete_all_indexes()

    async def get_index_stats(self) -> dict[str, Any]:
        """
        Get statistics for all indexes.

        Returns:
            dict with stats per index (documents, size)
        """
        stats = await self.opensearch.get_all_index_stats()
        return {
            'status': 'success',
            'indexes': stats,
        }

    async def ping_opensearch(self) -> bool:
        """Check if OpenSearch is reachable."""
        return await self.opensearch.ping()
