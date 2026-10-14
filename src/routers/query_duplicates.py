"""
Duplicate group routes of the data retrieval API.

- GET /query/duplicates - List duplicate groups
- GET /query/duplicates/stats - Duplicate detection statistics
- GET /query/duplicates/{group_id} - Get duplicate group members
"""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query

from src.core.dependencies import VisualSearchDep
from src.schemas.query import (
    DuplicateGroupDetailResponse,
    DuplicateGroupMember,
    DuplicateGroupsResponse,
    DuplicateGroupSummary,
    DuplicateStatsResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get('/duplicates', response_model=DuplicateGroupsResponse)
async def list_duplicate_groups(
    search_service: VisualSearchDep,
    page: Annotated[int, Query(ge=0, description='Page number (0-indexed)')] = 0,
    size: Annotated[int, Query(ge=1, le=100, description='Page size')] = 50,
    min_size: Annotated[int, Query(ge=2, description='Minimum group size')] = 2,
):
    """
    List duplicate groups.

    Returns groups of near-duplicate images detected during ingestion.
    Each group has a primary image (best quality) and duplicates.

    Args:
        page: Page number (0-indexed)
        size: Page size (max 100)
        min_size: Minimum group size to include (default: 2)

    Returns:
        Paginated list of duplicate groups.
    """
    try:
        from src.services.duplicate_detection import DuplicateDetectionService

        dup_service = DuplicateDetectionService(search_service.opensearch)

        # Get groups
        groups = await dup_service.get_duplicate_groups(
            min_size=min_size,
            page=page,
            size=size,
        )

        # Get stats
        stats = await dup_service.get_stats()

        group_summaries = [
            DuplicateGroupSummary(
                group_id=g.group_id,
                primary_image_id=g.primary_image_id,
                primary_image_path=g.primary_image_path,
                member_count=g.member_count,
            )
            for g in groups
        ]

        return DuplicateGroupsResponse(
            status='success',
            total_groups=stats.get('duplicate_groups', 0),
            total_duplicates=stats.get('grouped_images', 0),
            page=page,
            size=size,
            groups=group_summaries,
        )

    except Exception as e:
        logger.error(f'List duplicates failed: {e}')
        raise HTTPException(status_code=500, detail=f'Query failed: {e!s}') from e


@router.get('/duplicates/stats', response_model=DuplicateStatsResponse)
async def get_duplicate_stats(
    search_service: VisualSearchDep,
):
    """
    Get duplicate detection statistics.

    Returns overall statistics about duplicate groups:
    - Total images and how many are grouped
    - Number of duplicate groups
    - Average group size

    Returns:
        Duplicate detection statistics.
    """
    try:
        from src.services.duplicate_detection import DuplicateDetectionService

        dup_service = DuplicateDetectionService(search_service.opensearch)
        stats = await dup_service.get_stats()

        return DuplicateStatsResponse(
            status='success',
            total_images=stats.get('total_images', 0),
            grouped_images=stats.get('grouped_images', 0),
            ungrouped_images=stats.get('ungrouped_images', 0),
            duplicate_groups=stats.get('duplicate_groups', 0),
            average_group_size=stats.get('average_group_size', 0),
        )

    except Exception as e:
        logger.error(f'Get duplicate stats failed: {e}')
        raise HTTPException(status_code=500, detail=f'Stats retrieval failed: {e!s}') from e


@router.get('/duplicates/{group_id}', response_model=DuplicateGroupDetailResponse)
async def get_duplicate_group(
    search_service: VisualSearchDep,
    group_id: Annotated[str, Path(description='Duplicate group ID')],
):
    """
    Get members of a duplicate group.

    Returns all images in a duplicate group, sorted by:
    1. Primary first
    2. Then by similarity score (highest first)

    Args:
        group_id: Duplicate group ID

    Returns:
        Group members with metadata.
    """
    try:
        from src.services.duplicate_detection import DuplicateDetectionService

        dup_service = DuplicateDetectionService(search_service.opensearch)
        members = await dup_service.get_group_members(group_id, include_primary=True)

        if not members:
            return DuplicateGroupDetailResponse(
                status='not_found',
                group_id=group_id,
                error=f'Duplicate group {group_id} not found',
            )

        member_list = [
            DuplicateGroupMember(
                image_id=m.get('image_id', ''),
                image_path=m.get('image_path'),
                is_primary=m.get('is_duplicate_primary', False),
                duplicate_score=m.get('duplicate_score'),
                width=m.get('width'),
                height=m.get('height'),
                indexed_at=m.get('indexed_at'),
            )
            for m in members
        ]

        return DuplicateGroupDetailResponse(
            status='success',
            group_id=group_id,
            member_count=len(member_list),
            members=member_list,
        )

    except Exception as e:
        logger.error(f'Get duplicate group failed for {group_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Query failed: {e!s}') from e
