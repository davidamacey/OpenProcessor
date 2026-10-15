"""
Near-Duplicate Detection Service

Finds and groups visually similar images using CLIP embeddings.

Features:
- Find near-duplicates for a single image (k-NN search)
- Scan entire index to create duplicate groups
- Query duplicate groups
- Merge/unmerge duplicate groups

Algorithm:
1. For each ungrouped image, find k-NN neighbors with similarity > threshold
2. If neighbors found, create a group with image as primary
3. Mark all neighbors as duplicates of the primary
4. Primary is the first image ingested (or highest quality)

Usage:
    service = DuplicateDetectionService(opensearch_client)

    # Find near-duplicates for an image (default threshold=0.99, matches Immich)
    duplicates = await service.find_duplicates(image_id)

    # Use lower threshold to find more similar (not identical) images
    duplicates = await service.find_duplicates(image_id, threshold=0.95)

    # Get duplicate groups
    groups = await service.get_duplicate_groups(min_size=2)
"""

import logging
import uuid
from typing import Any

import numpy as np

from src.clients.opensearch.client import OpenSearchClient
from src.clients.opensearch.names import IndexName
from src.services.duplicate_detection.models import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DuplicateGroup,
    DuplicateMatch,
)
from src.services.duplicate_detection.queries import (
    GROUP_CARDINALITY_BODY,
    GROUPED_COUNT_BODY,
    group_from_bucket,
    group_member_doc,
    group_members_body,
    groups_aggregation_body,
    knn_body,
    matches_from_hits,
)


logger = logging.getLogger(__name__)


class DuplicateDetectionService:
    """
    Service for detecting and managing near-duplicate images.

    Uses CLIP embeddings stored in OpenSearch to find visually similar images
    via k-NN search. Groups duplicates together for easy management.
    """

    def __init__(self, opensearch: OpenSearchClient):
        """
        Initialize duplicate detection service.

        Args:
            opensearch: OpenSearch client for vector search
        """
        self.opensearch = opensearch

    async def find_duplicates(
        self,
        image_id: str,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_results: int = 50,
    ) -> list[DuplicateMatch]:
        """
        Find near-duplicates for a specific image.

        Args:
            image_id: Image to find duplicates for
            threshold: Minimum similarity score (0.0 - 1.0)
            max_results: Maximum number of duplicates to return

        Returns:
            List of DuplicateMatch objects sorted by similarity (highest first)
        """
        # Get the image embedding
        try:
            doc = await self.opensearch.client.get(
                index=IndexName.GLOBAL.value,
                id=image_id,
                _source=['global_embedding', 'image_path'],
            )
        except Exception as e:
            logger.error(f'Image not found: {image_id} - {e}')
            return []

        embedding = doc['_source']['global_embedding']

        # k-NN search for similar images
        response = await self.opensearch.client.search(
            index=IndexName.GLOBAL.value,
            body=knn_body(embedding, max_results + 1),  # +1 to exclude self
        )

        return matches_from_hits(response['hits']['hits'], threshold, image_id)

    async def find_duplicates_by_embedding(
        self,
        embedding: np.ndarray | list[float],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_results: int = 50,
        exclude_image_id: str | None = None,
    ) -> list[DuplicateMatch]:
        """
        Find near-duplicates using a raw embedding vector.

        Useful for checking duplicates before ingestion.

        Args:
            embedding: CLIP embedding vector (512-dim)
            threshold: Minimum similarity score
            max_results: Maximum results
            exclude_image_id: Image ID to exclude from results

        Returns:
            List of DuplicateMatch objects
        """
        vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        response = await self.opensearch.client.search(
            index=IndexName.GLOBAL.value,
            body=knn_body(vector, max_results),
        )

        return matches_from_hits(response['hits']['hits'], threshold, exclude_image_id)

    async def create_duplicate_group(
        self,
        primary_image_id: str,
        duplicate_image_ids: list[str],
        duplicate_scores: list[float],
    ) -> str:
        """
        Create a duplicate group with a primary image and duplicates.

        Args:
            primary_image_id: The "best" image in the group
            duplicate_image_ids: List of duplicate image IDs
            duplicate_scores: Similarity scores for each duplicate

        Returns:
            The new group ID
        """
        group_id = f'dup_{uuid.uuid4().hex[:12]}'

        # Update primary image
        await self.opensearch.client.update(
            index=IndexName.GLOBAL.value,
            id=primary_image_id,
            body=group_member_doc(group_id, primary=True, score=1.0),
        )

        # Update duplicate images
        for dup_id, score in zip(duplicate_image_ids, duplicate_scores, strict=False):
            await self.opensearch.client.update(
                index=IndexName.GLOBAL.value,
                id=dup_id,
                body=group_member_doc(group_id, primary=False, score=score),
            )

        logger.info(
            f'Created duplicate group {group_id} with {len(duplicate_image_ids) + 1} images'
        )
        return group_id

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
