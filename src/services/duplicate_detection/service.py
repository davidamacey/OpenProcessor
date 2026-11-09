"""Core duplicate group queries: the groups the single-writer ``near_dup_core_group`` job wrote.

Grouping is never done here or in the ingest request: core ingest writes pending rows and the job
groups them (``src/services/near_duplicates/core_group.py``). This service only reads the
projection of the current run from the global index.
"""

from typing import Any

from src.clients.opensearch.client import OpenSearchClient
from src.clients.opensearch.names import IndexName
from src.services.duplicate_detection.models import DuplicateGroup
from src.services.duplicate_detection.queries import (
    GROUP_CARDINALITY_BODY,
    GROUPED_COUNT_BODY,
    group_from_bucket,
    group_members_body,
    groups_aggregation_body,
)


class DuplicateDetectionService:
    """
    Read side of the core near-duplicate groups.
    """

    def __init__(self, opensearch: OpenSearchClient):
        """
        Initialize duplicate detection service.

        Args:
            opensearch: OpenSearch client for vector search
        """
        self.opensearch = opensearch

    async def get_duplicate_groups(
        self,
        min_size: int = 2,
        page: int = 0,
        size: int = 50,
    ) -> list[DuplicateGroup]:
        """
        Get all duplicate groups.

        Args:
            min_size: Minimum group size to include
            page: Page number (0-indexed)
            size: Page size

        Returns:
            List of DuplicateGroup objects
        """
        # Aggregate by duplicate_group_id
        response = await self.opensearch.client.search(
            index=IndexName.GLOBAL.value,
            body=groups_aggregation_body(min_size),
        )

        buckets = response['aggregations']['groups']['buckets']

        # Apply pagination
        start = page * size
        end = start + size
        return [group_from_bucket(bucket) for bucket in buckets[start:end]]

    async def get_group_members(
        self,
        group_id: str,
        include_primary: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get all members of a duplicate group.

        Args:
            group_id: Duplicate group ID
            include_primary: Whether to include the primary image

        Returns:
            List of member documents
        """
        response = await self.opensearch.client.search(
            index=IndexName.GLOBAL.value,
            body=group_members_body(group_id, include_primary),
        )

        return [hit['_source'] for hit in response['hits']['hits']]

    async def get_stats(self) -> dict[str, Any]:
        """
        Get duplicate detection statistics.

        Returns:
            Dict with counts and statistics
        """
        # Total images
        total = await self.opensearch.client.count(index=IndexName.GLOBAL.value)

        # Grouped images
        grouped = await self.opensearch.client.count(
            index=IndexName.GLOBAL.value,
            body=GROUPED_COUNT_BODY,
        )

        # Number of groups
        groups_response = await self.opensearch.client.search(
            index=IndexName.GLOBAL.value,
            body=GROUP_CARDINALITY_BODY,
        )

        num_groups = groups_response['aggregations']['group_count']['value']

        return {
            'total_images': total['count'],
            'grouped_images': grouped['count'],
            'ungrouped_images': total['count'] - grouped['count'],
            'duplicate_groups': num_groups,
            'average_group_size': grouped['count'] / num_groups if num_groups > 0 else 0,
        }
