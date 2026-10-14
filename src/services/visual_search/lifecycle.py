"""Index lifecycle management."""

import logging
from typing import Any

from src.clients.opensearch.client import OpenSearchClient
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class IndexLifecycleMixin:
    opensearch: OpenSearchClient
    inference: InferenceService

    # =========================================================================
    # Index Management (All Category Indexes)
    # =========================================================================

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
