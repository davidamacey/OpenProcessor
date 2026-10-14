"""
Data Retrieval Router - Query stored image data and metadata.

Provides endpoints for retrieving indexed data, statistics, and managing
stored images. The duplicate group routes live in ``query_duplicates`` and are
included last, as they were always registered.

Endpoints:
- GET /query/image/{image_id} - Get stored image data/metadata
- GET /query/stats - Get index statistics for all indexes
- DELETE /query/image/{image_id} - Remove image from all indexes
- GET /query/duplicates - List duplicate groups
- GET /query/duplicates/{group_id} - Get duplicate group members
"""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query

from src.core.dependencies import VisualSearchDep
from src.routers import query_duplicates
from src.schemas.query import (
    AllIndexStatsResponse,
    DeleteImageResponse,
    ImageQueryResponse,
    IndexStats,
)
from src.services.image_admin import delete_image_everywhere, fetch_image_data
from src.utils.upstream_errors import UpstreamError


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/query',
    tags=['Data Retrieval'],
)


# =============================================================================
# Helper Functions
# =============================================================================


def format_bytes(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    size: float = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size) < 1024.0:
            return f'{size:.1f} {unit}'
        size /= 1024.0
    return f'{size:.1f} PB'


# =============================================================================
# Endpoints
# =============================================================================


@router.get('/stats', response_model=AllIndexStatsResponse)
async def get_all_stats(
    search_service: VisualSearchDep,
):
    """
    Get statistics for all visual search indexes.

    Returns document counts and sizes for:
    - visual_search_global (whole image embeddings)
    - visual_search_vehicles (vehicle detections)
    - visual_search_people (person detections)
    - visual_search_faces (face embeddings)
    - visual_search_ocr (text content)

    Returns:
        Statistics for all indexes.
    """
    try:
        result = await search_service.get_index_stats()

        if result.get('status') == 'error':
            return AllIndexStatsResponse(
                status='error',
                error=result.get('error'),
            )

        indexes_data = result.get('indexes', {})
        indexes = []
        total_docs = 0
        total_size = 0

        for name, stats in indexes_data.items():
            doc_count = stats.get('doc_count', 0)
            size_bytes = stats.get('size_bytes', 0)
            total_docs += doc_count
            total_size += size_bytes

            indexes.append(
                IndexStats(
                    name=name,
                    doc_count=doc_count,
                    size_bytes=size_bytes,
                    size_human=format_bytes(size_bytes),
                    exists=stats.get('exists', True),
                )
            )

        return AllIndexStatsResponse(
            status='success',
            indexes=indexes,
            total_documents=total_docs,
            total_size_bytes=total_size,
            total_size_human=format_bytes(total_size),
        )

    except Exception as e:
        logger.error(f'Get stats failed: {e}')
        raise HTTPException(status_code=500, detail=f'Stats retrieval failed: {e!s}') from e


@router.get('/image/{image_id}', response_model=ImageQueryResponse)
async def get_image_data(
    search_service: VisualSearchDep,
    image_id: Annotated[str, Path(description='Image identifier')],
    include_embedding: Annotated[bool, Query(description='Include 512-dim CLIP embedding')] = False,
):
    """
    Get stored data and metadata for an image.

    Retrieves all indexed data for an image across all indexes:
    - Global metadata (dimensions, hash, timestamps)
    - Detection counts (objects, faces, vehicles, people)
    - Clustering info (cluster_id, distance)
    - Duplicate group info

    Args:
        image_id: Image identifier
        include_embedding: Include the 512-dim CLIP embedding

    Returns:
        Image metadata and indexed data summary.
    """
    try:
        return await fetch_image_data(search_service.opensearch.client, image_id, include_embedding)

    except (HTTPException, UpstreamError):
        raise
    except Exception as e:
        logger.error(f'Get image data failed for {image_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Query failed: {e!s}') from e


@router.delete('/image/{image_id}', response_model=DeleteImageResponse)
async def delete_image(
    search_service: VisualSearchDep,
    image_id: Annotated[str, Path(description='Image identifier to delete')],
):
    """
    Remove image from all indexes.

    Deletes the image and all associated data from:
    - visual_search_global
    - visual_search_vehicles (all detections)
    - visual_search_people (all detections)
    - visual_search_faces (all faces)
    - visual_search_ocr

    This is a permanent operation.

    Args:
        image_id: Image identifier to delete

    Returns:
        Deletion summary.
    """
    try:
        return await delete_image_everywhere(search_service.opensearch.client, image_id)

    except Exception as e:
        logger.error(f'Delete image failed for {image_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Deletion failed: {e!s}') from e


router.include_router(query_duplicates.router)
